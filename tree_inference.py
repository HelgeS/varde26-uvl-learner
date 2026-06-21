"""Tree inference, validation, cleanup, and refinement for learned constraint sets.

This module contains all tree-related post-processing logic extracted from
ca_uvl_notree.py:

- infer_and_refine_tree() — unified pipeline (replaces refine_completeness +
  infer_tree + refine_tree_groups)
- infer_tree() — pure tree inference from learned constraints (deprecated but used)
- constraints_from_tree() — reconstruct CPMpy constraints from tree structure
- cleanup_dumb() / cleanup_constraints() — remove spurious constraints
- _validate_tree() / _fix_multi_parent_tree() — tree validation and fixes
- refine_completeness() / refine_tree_groups() — deprecated, kept for compat
"""

import cpmpy as cp
from cpmpy.expressions.variables import _BoolVarImpl, NegBoolView
from cpmpy.expressions.core import Operator, Comparison
from pycona import ConstraintOracle


# ── Helper: extract feature names mentioned in a constraint ───────


def _mentioned_features(c) -> set[str]:
    """Return set of feature names that appear in constraint c."""
    names = set()

    def _walk(expr):
        if isinstance(expr, NegBoolView):
            names.add(expr._bv.name)
        elif isinstance(expr, _BoolVarImpl):
            names.add(expr.name)
        elif hasattr(expr, "args"):
            for arg in expr.args:
                _walk(arg)

    _walk(c)
    return names


def merge_shared_sets(sets):
    merged = []

    for s in sets:
        # Convert to a standard, mutable set
        current_set = set(s)
        unmerged = []

        # Compare the current set with the ones we've already processed
        for m in merged:
            if not current_set.isdisjoint(m):
                # If they share an item, combine them
                current_set.update(m)
            else:
                # If they don't share an item, keep the processed set as is
                unmerged.append(m)

        # Add the newly merged set back into the list
        unmerged.append(current_set)
        merged = unmerged

    return merged


# ── Unified tree inference + refinement ──────────────────────────────


def infer_and_refine_tree(
    feature_names: list[str],
    variables: list,
    learned_cl: list,
    oracle: ConstraintOracle,
) -> tuple[list, dict, int]:
    """Infer tree structure from learned constraints and refine with oracle queries.

    Combines the work of refine_completeness, infer_tree, and refine_tree_groups
    into a single pass.  Constraint parsing and tree topology are built once (pure),
    then each parent's groups are classified bottom-up using targeted oracle queries
    that skip when the answer is already implied by learned constraints.

    Returns ``(enhanced_cl, tree_info, n_queries)`` where:
    - enhanced_cl: learned_cl + any constraints added during refinement
    - tree_info: {parent: [(group_type, [children]), ...]}
    - n_queries: total oracle queries used
    """
    var_of = {name: v for name, v in zip(feature_names, variables)}
    n_queries = 0

    def _feat_name(expr):
        if isinstance(expr, NegBoolView):
            return expr._bv.name
        if isinstance(expr, _BoolVarImpl):
            return expr.name
        return None

    def _is_neg(expr):
        return isinstance(expr, NegBoolView)

    # ══════════════════════════════════════════════════════════════════
    # Phase 1: Parse learned constraints (pure)
    # ══════════════════════════════════════════════════════════════════
    impl: dict[str, set[str]] = {}
    excl: set[frozenset] = set()
    eq: set[frozenset] = set()
    core: set[str] = set()
    dead: set[str] = set()
    completeness: dict[str, list[set[str]]] = {}
    exactly_one: dict[str, list[set[str]]] = {}

    for c in learned_cl:
        if isinstance(c, Operator) and c.name == "->":
            lhs, rhs = c.args
            a = _feat_name(lhs)
            if a is None:
                continue
            b = _feat_name(rhs)
            if b is not None:
                if _is_neg(rhs):
                    excl.add(frozenset({a, b}))
                else:
                    impl.setdefault(a, set()).add(b)
            elif isinstance(rhs, Operator) and rhs.name == "or":
                members = set()
                for arg in rhs.args:
                    name = _feat_name(arg)
                    if name is not None and not _is_neg(arg):
                        members.add(name)
                if members:
                    completeness.setdefault(a, []).append(members)
            elif isinstance(rhs, Comparison) and rhs.name == "==":
                sum_expr, val = rhs.args
                if hasattr(val, "value"):
                    val = val.value()
                if val == 1 and hasattr(sum_expr, "args"):
                    members = set()
                    for arg in sum_expr.args:
                        name = _feat_name(arg)
                        if name is not None and not _is_neg(arg):
                            members.add(name)
                    if len(members) >= 2:
                        exactly_one.setdefault(a, []).append(members)
            continue
        if isinstance(c, Comparison) and c.name == "==":
            a_expr, b_expr = c.args
            a = _feat_name(a_expr)
            b = _feat_name(b_expr)
            if (
                a is not None
                and b is not None
                and not _is_neg(a_expr)
                and not _is_neg(b_expr)
            ):
                eq.add(frozenset({a, b}))
            continue
        if isinstance(c, NegBoolView):
            dead.add(c._bv.name)
            continue
        if isinstance(c, _BoolVarImpl):
            core.add(c.name)
            continue

    # Inject equivalence edges into impl
    for pair in eq:
        a, b = tuple(pair)
        impl.setdefault(a, set()).add(b)
        impl.setdefault(b, set()).add(a)

    # Infer implied edges from completeness/exactly-one
    for p, clause_sets in list(completeness.items()) + list(exactly_one.items()):
        for members in clause_sets:
            common_targets = None
            for m in members:
                m_targets = impl.get(m, set())
                if common_targets is None:
                    common_targets = set(m_targets)
                else:
                    common_targets &= m_targets
            if common_targets:
                for q in common_targets:
                    if q != p:
                        impl.setdefault(p, set()).add(q)

    for p, hint_sets in exactly_one.items():
        for members in hint_sets:
            for m in members:
                impl.setdefault(m, set()).add(p)

    # Snapshot of impl before tree topology modifies direct_impl.
    # Used later in completeness recovery (step 3.0b) so that ALL
    # learned Ci => P edges are considered, not just surviving tree edges.
    orig_impl = {a: set(bs) for a, bs in impl.items()}

    # ══════════════════════════════════════════════════════════════════
    # Phase 2: Build tree topology (pure)
    # ══════════════════════════════════════════════════════════════════

    # Find root
    in_degree: dict[str, int] = {f: 0 for f in feature_names}
    for src, targets in impl.items():
        for tgt in targets:
            if tgt in in_degree:
                in_degree[tgt] += 1

    root = None
    for f in feature_names:
        if f in core and in_degree[f] == 0:
            root = f
            break
    if root is None:
        reach_count: dict[str, int] = {}
        for f in feature_names:
            visited: set[str] = set()
            queue = [f]
            while queue:
                node = queue.pop()
                if node in visited:
                    continue
                visited.add(node)
                for src, targets in impl.items():
                    if node in targets and src not in visited:
                        queue.append(src)
            reach_count[f] = len(visited) - 1
        root = max(feature_names, key=lambda f: reach_count[f])
    if root is None:
        root = feature_names[0]

    # Break 2-cycles from equivalences
    direct_impl: dict[str, set[str]] = {a: set(bs) for a, bs in impl.items()}

    for pair in eq:
        a, b = tuple(pair)
        a_to_b = b in direct_impl.get(a, set())
        b_to_a = a in direct_impl.get(b, set())
        if a_to_b and b_to_a:
            a_eq = sum(1 for p in eq if a in p)
            b_eq = sum(1 for p in eq if b in p)
            if a_eq != b_eq:
                if a_eq > b_eq:
                    direct_impl[a].discard(b)
                else:
                    direct_impl[b].discard(a)
                continue
            a_out = len(direct_impl.get(a, set()) - {b})
            b_out = len(direct_impl.get(b, set()) - {a})
            if a_out != b_out:
                if a_out > b_out:
                    direct_impl[a].discard(b)
                else:
                    direct_impl[b].discard(a)
                continue
            a_excl = sum(1 for p in excl if a in p)
            b_excl = sum(1 for p in excl if b in p)
            if a_excl != b_excl:
                if a_excl > b_excl:
                    direct_impl[a].discard(b)
                else:
                    direct_impl[b].discard(a)
                continue
            if in_degree.get(a, 0) >= in_degree.get(b, 0):
                direct_impl[a].discard(b)
            else:
                direct_impl[b].discard(a)

    direct_impl.get(root, set()).clear()

    # Transitive reduction
    edges_to_remove: set[tuple[str, str]] = set()
    for a in list(direct_impl.keys()):
        for b in list(direct_impl.get(a, set())):
            visited_tr: set[str] = set()
            queue_tr = list(direct_impl.get(a, set()) - {b})
            found = False
            while queue_tr and not found:
                node = queue_tr.pop()
                if node == b:
                    found = True
                    break
                if node in visited_tr:
                    continue
                visited_tr.add(node)
                queue_tr.extend(direct_impl.get(node, set()))
            if found:
                edges_to_remove.add((a, b))
    for a, b in edges_to_remove:
        direct_impl[a].discard(b)

    # Redirect edges through equivalences
    for pair in eq:
        a, b = tuple(pair)
        a_to_b = b in direct_impl.get(a, set())
        b_to_a = a in direct_impl.get(b, set())
        if b_to_a and not a_to_b:
            parent_eq, child_eq = a, b
        elif a_to_b and not b_to_a:
            parent_eq, child_eq = b, a
        else:
            continue
        child_members: set[str] = set()
        for members in completeness.get(child_eq, []):
            child_members.update(members)
        for members in exactly_one.get(child_eq, []):
            child_members.update(members)
        for members in completeness.get(parent_eq, []):
            child_members.update(members)
        for members in exactly_one.get(parent_eq, []):
            child_members.update(members)
        if not child_members:
            child_children = {
                x
                for x, targets in direct_impl.items()
                if child_eq in targets and x != parent_eq
            }
            if child_children:
                for x, targets in list(direct_impl.items()):
                    if parent_eq in targets and x != child_eq:
                        has_excl = any(
                            frozenset({x, cc}) in excl for cc in child_children
                        )
                        if has_excl:
                            child_members.add(x)
        if not child_members:
            continue
        for x in child_members:
            x_targets = direct_impl.get(x, set())
            if parent_eq in x_targets:
                x_targets.discard(parent_eq)
                x_targets.add(child_eq)

    # Multi-parent resolution via co-sibling scoring
    all_linked: set[frozenset] = set()
    for a_name, targets in impl.items():
        for b_name in targets:
            all_linked.add(frozenset({a_name, b_name}))
    all_linked |= excl | eq

    edges_to_prune: list[tuple[str, str]] = []
    for x in direct_impl:
        targets = direct_impl.get(x, set())
        if len(targets) <= 1:
            continue
        scores: dict[str, int] = {}
        for p in targets:
            co_children = {
                other
                for other, other_targets in direct_impl.items()
                if other != x and p in other_targets
            }
            scores[p] = sum(
                1 for sib in co_children if frozenset({x, sib}) in all_linked
            )
        best_score = max(scores.values())
        if best_score > 0:
            best_parents = {p for p, s in scores.items() if s == best_score}
            for p in targets:
                if p not in best_parents:
                    edges_to_prune.append((x, p))
    for x, p in edges_to_prune:
        direct_impl.get(x, set()).discard(p)

    # Break remaining cycles
    def _find_cycle(graph: dict[str, set[str]], start: str) -> list[str] | None:
        visited_c, stack = set(), [(start, [start])]
        while stack:
            node, path = stack.pop()
            if node in visited_c:
                continue
            visited_c.add(node)
            for nxt in graph.get(node, set()):
                if nxt == start:
                    return path + [nxt]
                if nxt not in visited_c:
                    stack.append((nxt, path + [nxt]))
        return None

    changed = True
    max_iters = len(feature_names)
    while changed and max_iters > 0:
        changed = False
        max_iters -= 1
        for node in list(direct_impl.keys()):
            cycle = _find_cycle(direct_impl, node)
            if cycle and len(cycle) > 1:
                weakest_edge = None
                weakest_score = float("inf")
                for ci in range(len(cycle) - 1):
                    a, b = cycle[ci], cycle[ci + 1]
                    co_children = {
                        other
                        for other, other_targets in direct_impl.items()
                        if other != a and b in other_targets
                    }
                    score = sum(
                        1 for sib in co_children if frozenset({a, sib}) in all_linked
                    )
                    if score < weakest_score:
                        weakest_score = score
                        weakest_edge = (a, b)
                if weakest_edge:
                    direct_impl.get(weakest_edge[0], set()).discard(weakest_edge[1])
                    changed = True
                    break

    # Resolve remaining multi-parent edges
    for x in list(direct_impl.keys()):
        targets = direct_impl.get(x, set())
        if len(targets) <= 1:
            continue
        scores: dict[str, int] = {}
        for p in targets:
            co_children = {
                other
                for other, other_targets in direct_impl.items()
                if other != x and p in other_targets
            }
            scores[p] = sum(
                1 for sib in co_children if frozenset({x, sib}) in all_linked
            )
        best_score = max(scores.values())
        best_parents = {p for p, s in scores.items() if s == best_score}
        if len(best_parents) > 1:
            child_counts = {}
            for p in best_parents:
                child_counts[p] = sum(
                    1
                    for other, other_targets in direct_impl.items()
                    if p in other_targets
                )
            best_count = max(child_counts.values())
            best_parents = {p for p, c in child_counts.items() if c == best_count}
        keep = next(iter(best_parents))
        for p in list(targets):
            if p != keep:
                targets.discard(p)

    # Build children_of from direct_impl
    children_of: dict[str, set[str]] = {f: set() for f in feature_names}
    for a, targets in direct_impl.items():
        for p in targets:
            if p in children_of:
                children_of[p].add(a)

    print(f"  infer_and_refine: tree topology built, root={root}")

    # ══════════════════════════════════════════════════════════════════
    # Phase 3: Classify groups with oracle (bottom-up)
    # ══════════════════════════════════════════════════════════════════

    added_cl: list = []
    structural_cl: list = []

    # Build ancestor map
    tree_edges: set[tuple[str, str]] = set()  # (child, parent)
    for parent_name in feature_names:
        for child_name in children_of[parent_name]:
            tree_edges.add((child_name, parent_name))

    ancestors: dict[str, set[str]] = {f: set() for f in feature_names}
    for child_name, parent_name in tree_edges:
        ancestors[child_name].add(parent_name)
    changed = True
    while changed:
        changed = False
        for child_name in feature_names:
            for parent_name in list(ancestors[child_name]):
                new_anc = ancestors[parent_name] - ancestors[child_name]
                if new_anc:
                    ancestors[child_name].update(new_anc)
                    changed = True

    def _is_excl_between(c, pairs: set[frozenset]) -> bool:
        if not (isinstance(c, Operator) and c.name == "->"):
            return False
        lhs, rhs = c.args
        a = (
            lhs.name
            if isinstance(lhs, _BoolVarImpl) and not isinstance(lhs, NegBoolView)
            else None
        )
        if a is None:
            return False
        if isinstance(rhs, NegBoolView):
            b = rhs._bv.name
            return frozenset({a, b}) in pairs
        return False

    def _ask(
        assumptions: list,
        *,
        exclude_excl: set[frozenset] | None = None,
        retries: int = 0,
    ) -> bool | None:
        nonlocal n_queries
        if exclude_excl:
            base_cl = [c for c in learned_cl if not _is_excl_between(c, exclude_excl)]
            base_cl += [
                c for c in structural_cl if not _is_excl_between(c, exclude_excl)
            ]
        else:
            base_cl = list(learned_cl) + list(structural_cl)
        base_cl = base_cl + added_cl

        feats = []
        for c in assumptions:
            feats.extend(_mentioned_features(c))
        reach = set()
        for f in feats:
            reach.update(ancestors.get(f, set()))
        reachability = [var_of[a] for a in reach if a in var_of]

        blocking: list = []
        for attempt in range(1 + retries):
            m = cp.Model(base_cl + assumptions + reachability + blocking)
            # TODO This is a cheap heuristic to avoid the case where not all group constraints are known yet
            # Unfortunately, it does not work reliably
            m.maximize(cp.sum(variables))
            if not m.solve():
                return None if attempt == 0 else False
            n_queries += 1
            try:
                result = oracle.answer_membership_query(list(variables))
            except Exception as e:
                if "Collapse" in str(e):
                    return None
                raise
            if result is True or attempt >= retries:
                return result
            assignment = [v if v.value() else ~v for v in variables]
            blocking.append(~cp.all(assignment))

        return False

    # 3.0 Root core check
    root_var = var_of[root]
    ans = _ask([~root_var])
    if ans is None or ans is False:
        added_cl.append(root_var)
        structural_cl.append(root_var)
        print(f"  infer_and_refine: root '{root}' is core")

    # 3.0b Completeness recovery (like refine_completeness but integrated)
    # Binary CA cannot learn n-ary completeness clauses like P -> or(C1..Ck).
    # Recover them now using learned_cl only (no structural_cl pollution),
    # so that subsequent group classification sees them as hints.
    #
    # Use orig_impl (pre-tree-topology snapshot) so that ALL learned Ci => P
    # edges are considered. Tree topology steps (transitive reduction,
    # multi-parent resolution, cycle breaking) prune edges from direct_impl,
    # which would cause us to miss children and fail the completeness test.
    #
    # Exclude children that have equivalence with the parent (mandatory)
    # from the test — their presence would make UNSAT trivially true
    # even when the remaining children are all optional.
    impl_children: dict[str, list] = {f: [] for f in feature_names}
    for a, targets in orig_impl.items():
        for p in targets:
            if p in impl_children:
                impl_children[p].append(var_of[a])
    for parent_name, child_vars in impl_children.items():
        # Filter out mandatory (equivalence) children
        non_eq_cvs = [
            cv for cv in child_vars if frozenset({parent_name, cv.name}) not in eq
        ]
        if len(non_eq_cvs) < 2:
            continue
        parent_var = var_of[parent_name]
        assumptions = [parent_var] + [~cv for cv in non_eq_cvs]
        if not cp.Model(list(learned_cl) + assumptions).solve():
            continue  # already implied
        Y = [int(v.value()) if v.value() is not None else 0 for v in variables]
        n_queries += 1
        if not oracle.answer_membership_query(Y):
            new_c = parent_var.implies(cp.any(non_eq_cvs))
            added_cl.append(new_c)
            # Also add to completeness hints for Phase 3 grouping
            child_names = {cv.name for cv in non_eq_cvs}
            completeness.setdefault(parent_name, []).append(child_names)
            print(f"  infer_and_refine: recovered completeness {parent_name} => any({sorted(child_names)})")

    # 3.1 Bottom-up processing order
    depths: dict[str, int] = {}
    queue_bfs = [(root, 0)]
    while queue_bfs:
        node, d = queue_bfs.pop(0)
        if node in depths:
            continue
        depths[node] = d
        for child_name in children_of.get(node, set()):
            if child_name not in depths:
                queue_bfs.append((child_name, d + 1))
    processing_order = sorted(feature_names, key=lambda f: -depths.get(f, 0))

    tree_info: dict[str, list] = {}

    for parent in processing_order:
        children = list(children_of[parent])
        if not children:
            continue
        pv = var_of[parent]
        children_set = set(children)
        used: set[str] = set()
        groups: list[list] = []

        # 3a. Partition children into candidate sub-groups

        # exactly_one hint → candidate alternative group
        for hint_set in exactly_one.get(parent, []):
            matched = (children_set & hint_set) - used
            if len(matched) >= 2:
                matched_list = sorted(matched)
                # All pairs already in excl → known alternative (no queries needed)
                all_excl = all(
                    frozenset({matched_list[i], matched_list[j]}) in excl
                    for i in range(len(matched_list))
                    for j in range(i + 1, len(matched_list))
                )
                if all_excl:
                    # Completeness is implied by exactly_one → skip query
                    cvs = [var_of[c] for c in matched_list]
                    structural_cl.append(pv.implies(cp.any(cvs)))
                    for i in range(len(cvs)):
                        for j in range(i + 1, len(cvs)):
                            structural_cl.append(cvs[i].implies(~cvs[j]))
                    groups.append(["alternative", matched_list])
                    used.update(matched)
                    print(f"  infer_and_refine: {parent} → alternative {matched_list} (exactly_one+excl hint)")
                else:
                    # exactly_one hint but not all pairs excluded → test with oracle below
                    # Mark as candidate group for completeness testing
                    groups.append(["_candidate_exactly_one", matched_list])
                    used.update(matched)

        # completeness hint → candidate or/alt group
        parent_comp = completeness.get(parent, [])
        if parent_comp and (children_set - used):
            filtered = []
            for hint_set in parent_comp:
                subset = (hint_set & children_set) - used
                if len(subset) >= 2:
                    filtered.append(subset)
            if filtered:
                core_set = filtered[0]
                for h in filtered[1:]:
                    core_set = core_set & h
                if len(core_set) >= 2:
                    core_list = sorted(core_set)
                    all_excl = all(
                        frozenset({core_list[i], core_list[j]}) in excl
                        for i in range(len(core_list))
                        for j in range(i + 1, len(core_list))
                    )
                    if all_excl:
                        # Completeness is given by hint, all pairs excluded → alternative
                        cvs = [var_of[c] for c in core_list]
                        structural_cl.append(pv.implies(cp.any(cvs)))
                        for i in range(len(cvs)):
                            for j in range(i + 1, len(cvs)):
                                structural_cl.append(cvs[i].implies(~cvs[j]))
                        groups.append(["alternative", core_list])
                        used.update(core_set)
                        print(f"  infer_and_refine: {parent} → alternative {core_list} (completeness+excl hint)")
                    else:
                        groups.append(["_candidate_completeness", core_list])
                        used.update(core_set)

        # Mutual exclusion clique → candidate group
        remaining = [c for c in children if c not in used]
        if len(remaining) >= 2:
            excl_children = []
            for i in range(len(remaining)):
                for j in range(i + 1, len(remaining)):
                    cs = frozenset({remaining[i], remaining[j]})
                    if cs in excl:
                        excl_children.append(cs)
            alternative_sets = merge_shared_sets(excl_children)
            for alt_set in alternative_sets:
                alt_list = sorted(alt_set)
                groups.append(["_candidate_excl_clique", alt_list])
                used.update(alt_set)
                remaining = [c for c in remaining if c not in alt_set]

        # Remaining → individual singletons
        for c in remaining:
            groups.append(["_singleton", [c]])

        # 3b-3d. Test and classify each candidate group with oracle
        #
        # Two-pass approach:
        #   Pass 1: classify candidate multi-child groups (completeness + alt/or)
        #           and collect singletons.
        #   Pass 2: test completeness on collected singletons as a group,
        #           then classify remaining singletons as mandatory/optional.

        final_groups: list[list] = []
        pending_singletons: list[
            str
        ] = []  # singletons awaiting group completeness test

        # ── Pass 1: candidate groups and already-classified groups ──
        for gtype, ch in groups:
            cvs = [var_of[c] for c in ch]

            if gtype in ("alternative", "or"):
                # Already classified (from hints with all-excl) — keep as-is
                for cv in cvs:
                    structural_cl.append(cv.implies(pv))
                final_groups.append([gtype, ch])
                continue

            if gtype.startswith("_candidate"):
                # 3b. Test completeness
                has_eq_parent = any(frozenset({parent, c}) in eq for c in ch)

                if has_eq_parent and len(ch) == 1:
                    # Single child with equivalence → mandatory, not a group
                    structural_cl.append(pv.implies(cvs[0]))
                    structural_cl.append(cvs[0].implies(pv))
                    added_cl.append(pv.implies(cvs[0]))
                    final_groups.append(["mandatory", ch])
                    print(f"  infer_and_refine: {parent}/{ch[0]} → mandatory (eq)")
                    continue

                # SAT-test: can parent be true with all group children false?
                assumption = [pv] + [~cv for cv in cvs]
                comp_ans = _ask(assumption, retries=3)
                is_complete = comp_ans is None or comp_ans is False

                if not is_complete:
                    # Not complete → decompose to singletons for later processing
                    pending_singletons.extend(ch)
                    print(f"  infer_and_refine: {parent} candidate {ch} → not complete, decomposing")
                    continue

                # Complete group — now determine alt vs or (3c)
                child_pairs = {
                    frozenset({ch[ci], ch[cj]})
                    for ci in range(len(ch))
                    for cj in range(ci + 1, len(ch))
                }
                all_excl = all(p in excl for p in child_pairs)
                if all_excl:
                    gtype_final = "alternative"
                else:
                    found_coexist = False
                    for pi in range(len(cvs)):
                        for pj in range(pi + 1, len(cvs)):
                            pair_ans = _ask(
                                [pv, cvs[pi], cvs[pj]],
                                exclude_excl=child_pairs,
                                retries=3,
                            )
                            if pair_ans is True:
                                found_coexist = True
                                break
                        if found_coexist:
                            break
                    gtype_final = "or" if found_coexist else "alternative"

                # Add structural constraints
                structural_cl.append(pv.implies(cp.any(cvs)))
                added_cl.append(pv.implies(cp.any(cvs)))
                if gtype_final == "alternative":
                    for i in range(len(cvs)):
                        for j in range(i + 1, len(cvs)):
                            structural_cl.append(cvs[i].implies(~cvs[j]))
                            added_cl.append(cvs[i].implies(~cvs[j]))
                for cv in cvs:
                    structural_cl.append(cv.implies(pv))
                final_groups.append([gtype_final, ch])
                print(f"  infer_and_refine: {parent} → {gtype_final} {ch}")
                continue

            if gtype == "_singleton":
                pending_singletons.append(ch[0])
                continue

        # ── Pass 2: process collected singletons ──
        # First, pull out singletons that are mandatory by equivalence —
        # these must NOT participate in group completeness testing because
        # they make the UNSAT check trivially true.
        eq_mandatory: list[str] = []
        non_eq_singletons: list[str] = []
        for c in pending_singletons:
            if frozenset({parent, c}) in eq:
                eq_mandatory.append(c)
            else:
                non_eq_singletons.append(c)

        for c in eq_mandatory:
            cv = var_of[c]
            structural_cl.append(pv.implies(cv))
            structural_cl.append(cv.implies(pv))
            added_cl.append(pv.implies(cv))
            final_groups.append(["mandatory", [c]])
            print(f"  infer_and_refine: {parent}/{c} → mandatory (eq)")

        # Test group completeness on remaining (non-eq) singletons.
        # This prevents the false-mandatory problem: when a learned
        # completeness clause like P -> or(C1, C2) is present, testing
        # each child individually would make both appear mandatory.
        if len(non_eq_singletons) >= 2:
            sing_cvs = [var_of[c] for c in non_eq_singletons]
            assumption = [pv] + [~cv for cv in sing_cvs]
            comp_ans = _ask(assumption, retries=3)
            is_complete = comp_ans is None or comp_ans is False

            if is_complete:
                # Test alt vs or on the merged group
                child_pairs = {
                    frozenset({non_eq_singletons[i], non_eq_singletons[j]})
                    for i in range(len(non_eq_singletons))
                    for j in range(i + 1, len(non_eq_singletons))
                }
                all_excl = all(p in excl for p in child_pairs)
                if all_excl:
                    gtype_final = "alternative"
                else:
                    found_coexist = False
                    for pi in range(len(sing_cvs)):
                        for pj in range(pi + 1, len(sing_cvs)):
                            # TODO Validate for model_20120915_487659597.uvl
                            # Three or features but with extra exclusive constraint
                            pair_ans = _ask(
                                [pv, sing_cvs[pi], sing_cvs[pj]],
                                exclude_excl=child_pairs,
                                retries=3,
                            )
                            if pair_ans is True:
                                found_coexist = True
                                break
                        if found_coexist:
                            break
                    gtype_final = "or" if found_coexist else "alternative"

                structural_cl.append(pv.implies(cp.any(sing_cvs)))
                added_cl.append(pv.implies(cp.any(sing_cvs)))
                if gtype_final == "alternative":
                    for i in range(len(sing_cvs)):
                        for j in range(i + 1, len(sing_cvs)):
                            structural_cl.append(sing_cvs[i].implies(~sing_cvs[j]))
                            added_cl.append(sing_cvs[i].implies(~sing_cvs[j]))
                for cv in sing_cvs:
                    structural_cl.append(cv.implies(pv))
                final_groups.append([gtype_final, sorted(non_eq_singletons)])
                print(f"  infer_and_refine: {parent} → {gtype_final} {sorted(non_eq_singletons)} (from singletons)")
                non_eq_singletons = []  # all consumed

        # Process remaining singletons as mandatory/optional (3d)
        # (eq-mandatory singletons were already handled above)
        # Use retries to avoid false-mandatory from solver finding
        # assignments the oracle rejects for unrelated reasons.
        for c in non_eq_singletons:
            cv = var_of[c]
            ans = _ask([pv, ~cv], retries=3)
            if ans is None or ans is False:
                structural_cl.append(pv.implies(cv))
                structural_cl.append(cv.implies(pv))
                added_cl.append(pv.implies(cv))
                final_groups.append(["mandatory", [c]])
                print(f"  infer_and_refine: {parent}/{c} → mandatory")
            else:
                structural_cl.append(cv.implies(pv))
                final_groups.append(["optional", [c]])

        tree_info[parent] = final_groups

    # 3e. Verify parent-child edges
    for parent in list(tree_info.keys()):
        if parent == root:
            continue
        pv = var_of[parent]
        new_groups = []
        relocated: list[str] = []
        for gt, ch in tree_info[parent]:
            valid_children = []
            for c in ch:
                cv = var_of[c]
                ans = _ask([cv, ~pv])
                if ans is True:
                    relocated.append(c)
                    print(f"  infer_and_refine: {c} not child of {parent} (relocating to root)")
                else:
                    valid_children.append(c)
            if valid_children:
                new_groups.append([gt, valid_children])
        if relocated:
            tree_info[parent] = new_groups if new_groups else []
            if not tree_info[parent]:
                del tree_info[parent]
            root_groups = tree_info.setdefault(root, [])
            for c in relocated:
                root_groups.append(["optional", [c]])

    # 3.7 Attach unattached features as optional under root
    attached: set[str] = set()
    for groups in tree_info.values():
        for _, ch in groups:
            attached.update(ch)
    unattached = [f for f in feature_names if f != root and f not in attached]
    if unattached:
        tree_info.setdefault(root, []).append(["optional", unattached])

    # Convert to tuple format
    final_tree = {p: [(g[0], g[1]) for g in gs] for p, gs in tree_info.items()}

    enhanced_cl = list(learned_cl) + added_cl
    print(f"  infer_and_refine: +{len(added_cl)} constraint(s), {n_queries} oracle queries")
    return list(set(enhanced_cl)), final_tree, n_queries


# ── DEPRECATED: replaced by infer_and_refine_tree() ──────────────────
# The following three functions (refine_completeness, infer_tree,
# refine_tree_groups) are kept for backward compatibility and for use
# by HintingCAEnv._refresh_structural_hints() which only needs infer_tree.


def refine_completeness(
    feature_names: list[str],
    variables: list,
    learned_cl: list,
    oracle: ConstraintOracle,
) -> tuple[list, int, int]:
    """Add missing group-completeness clauses after binary CA converges.

    Binary CA can learn all pairwise constraints (A=>B, A=>~B, A==B) but
    cannot learn clauses with 3+ literals such as ``Sauce => (Ketchup | Mustard)``.
    These arise whenever a parent feature requires at least one of its children.

    Algorithm
    ---------
    1. Build implication graph from learned_cl: edge C→P for each ``C => P``.
    2. For each parent P with ≥2 children {C1..Ck}:
       - Solve ``learned_cl ∧ P ∧ ¬C1 ∧ … ∧ ¬Ck`` to find a candidate assignment.
       - If UNSAT → the clause is already implied by learned constraints — skip.
       - Otherwise ask the oracle: is this assignment a valid solution?
         If No → the target forbids P=True with all children False
              → append ``P => any([C1..Ck])`` to the returned list.
    3. Return (extended_cl, n_added, n_queries).

    One oracle query per candidate parent group — no enumeration of subsets.
    """
    var_of = {name: v for name, v in zip(feature_names, variables)}

    # Collect implication edges C => P (only plain positive BoolVars on both sides)
    children_of: dict[str, list] = {f: [] for f in feature_names}
    for c in learned_cl:
        if isinstance(c, Operator) and c.name == "->":
            lhs, rhs = c.args
            # Require plain (non-negated) BoolVars on both sides.
            # NegBoolView is a subclass of _BoolVarImpl, so check it explicitly.
            if (
                isinstance(lhs, _BoolVarImpl)
                and not isinstance(lhs, NegBoolView)
                and isinstance(rhs, _BoolVarImpl)
                and not isinstance(rhs, NegBoolView)
            ):
                children_of[rhs.name].append(lhs)  # lhs is child of rhs (parent)

    # TODO This can be a good query strategy after all binary constraints have been exhausted
    # Aggregate and then ask specific queries for group constraints

    added = []
    n_queries = 0
    for parent_name, children in children_of.items():
        if len(children) < 2:
            continue
        parent_var = var_of[parent_name]
        assumptions = [parent_var] + [~cv for cv in children]

        # Find a candidate assignment consistent with learned constraints + assumptions.
        if not cp.Model(learned_cl + assumptions).solve():
            # Already impossible in learned constraints — clause not needed.
            continue

        # Extract the assignment as an ordered value list and ask the oracle.
        Y = [int(v.value()) for v in variables]
        n_queries += 1
        if not oracle.answer_membership_query(Y):
            new_c = parent_var.implies(cp.any(children))
            added.append(new_c)
            print(f"  refine: {parent_name} => any({[cv.name for cv in children]})")
        # else:  # was (or)? both can be negative, then it's optional

    print(f"  completeness refinement: +{len(added)} clause(s) in {n_queries} queries")
    return list(learned_cl) + added, len(added), n_queries


# ── Tree inference from flat learned constraints ───────────────────


def infer_tree(feature_names: list[str], variables: list, learned_cl: list) -> dict:
    """Post-process a flat learned constraint list into a tree_info dict.

    The returned dict has the same format as extract_features() in ca_uvl.py:
        {parent_name: [(group_type, [child_names]), ...]}

    Algorithm
    ---------
    1.  Parse learned CL into typed sets (impl, excl, eq, core, dead) and
        completeness hints from A → or([B, C, ...]) clauses.
    1b. Inject equivalence edges: A == B adds A→B and B→A to impl.
    2.  Find root: core feature with no incoming implication edges; fallback to
        highest in-degree node; fallback to feature_names[0].
    3.  Transitive reduction of the implication graph to isolate direct edges.
    3b. Break 2-cycles from equivalences: keep the edge towards the node with
        more outgoing edges (more "parent-like"); tie-break by in-degree.
    3c. Redirect edges through equivalences: when B is child of A (from A==B)
        and B has completeness clauses claiming features X, redirect X→A to
        X→B so those features land under the correct subtree.
    4.  Group children by parent (A → P means A is a child of P).
    5.  Classify each (parent, children) group.
    6.  Attach unattached features as optional children of root.
    """
    var_name = {v.name: v for v in variables}

    # ── Step 1: parse learned CL ─────────────────────────────────────
    impl: dict[str, set[str]] = {}  # impl[A] = {B, ...}  (A => B)
    excl: set[frozenset] = set()  # {A, B} means A => ~B
    eq: set[frozenset] = set()  # {A, B} means A == B
    core: set[str] = set()  # unary True literals
    dead: set[str] = set()  # unary False literals
    # completeness[A] = [{B, C, ...}, ...] from A → or([B, C, ...]) clauses
    completeness: dict[str, list[set[str]]] = {}
    # exactly_one[A] = [{B, C, ...}, ...] from A → (sum([B, C, ...]) == 1)
    exactly_one: dict[str, list[set[str]]] = {}

    def _feat_name(expr):
        """Return feature name from a BoolVar or NegBoolView, else None."""
        # NegBoolView is a subclass of _BoolVarImpl — check it first.
        if isinstance(expr, NegBoolView):
            return expr._bv.name
        if isinstance(expr, _BoolVarImpl):
            return expr.name
        return None

    def _is_neg(expr):
        return isinstance(expr, NegBoolView)

    for c in learned_cl:
        # A => B  or  A => ~B  or  A => or([B, C, ...])
        if isinstance(c, Operator) and c.name == "->":
            lhs, rhs = c.args
            a = _feat_name(lhs)
            if a is None:
                continue
            b = _feat_name(rhs)
            if b is not None:
                if _is_neg(rhs):
                    excl.add(frozenset({a, b}))
                else:
                    impl.setdefault(a, set()).add(b)
            elif isinstance(rhs, Operator) and rhs.name == "or":
                # A → or([B, C, ...]) — group completeness hint
                members = set()
                for arg in rhs.args:
                    name = _feat_name(arg)
                    if name is not None and not _is_neg(arg):
                        members.add(name)
                if members:
                    completeness.setdefault(a, []).append(members)
            elif isinstance(rhs, Comparison) and rhs.name == "==":
                # A → (sum([B, C, ...]) == 1) — exactly-one hint
                sum_expr, val = rhs.args
                if hasattr(val, "value"):
                    val = val.value()
                if val == 1 and hasattr(sum_expr, "args"):
                    members = set()
                    for arg in sum_expr.args:
                        name = _feat_name(arg)
                        if name is not None and not _is_neg(arg):
                            members.add(name)
                    if len(members) >= 2:
                        exactly_one.setdefault(a, []).append(members)
            continue

        # A == B  (Comparison with ==)
        if isinstance(c, Comparison) and c.name == "==":
            a_expr, b_expr = c.args
            a = _feat_name(a_expr)
            b = _feat_name(b_expr)
            if (
                a is not None
                and b is not None
                and not _is_neg(a_expr)
                and not _is_neg(b_expr)
            ):
                eq.add(frozenset({a, b}))
            continue

        # Unary: plain BoolVar → core; NegBoolView → dead
        # NegBoolView is a subclass of _BoolVarImpl — check it first.
        if isinstance(c, NegBoolView):
            dead.add(c._bv.name)
            continue
        if isinstance(c, _BoolVarImpl):
            core.add(c.name)
            continue

    # ── Step 1b: inject equivalence edges into impl ────────────────────
    # A == B implies both A → B and B → A.  Without this, features that
    # *only* appear in equivalences (like Value in Fixed == Value) would
    # have zero implication edges and fall through to the "unattached"
    # catch-all in Step 6.
    for pair in eq:
        a, b = tuple(pair)
        impl.setdefault(a, set()).add(b)
        impl.setdefault(b, set()).add(a)

    # ── Step 1c: infer implied edges from completeness/exactly-one ─────
    # 1c-i: If P => or(C1..Ck) and every Ci => Q (Q ≠ P), then P => Q.
    #        Lets transitive reduction remove child-to-grandparent edges.
    for p, clause_sets in list(completeness.items()) + list(exactly_one.items()):
        for members in clause_sets:
            common_targets = None
            for m in members:
                m_targets = impl.get(m, set())
                if common_targets is None:
                    common_targets = set(m_targets)
                else:
                    common_targets &= m_targets
            if common_targets:
                for q in common_targets:
                    if q != p:
                        impl.setdefault(p, set()).add(q)

    # 1c-ii: If P => exactly_one(C1..Ck), each Ci should imply P.
    #         The CA may miss some Ci => P due to non-determinism; this
    #         recovers the missing child→parent edges for alternative groups.
    for p, hint_sets in exactly_one.items():
        for members in hint_sets:
            for m in members:
                impl.setdefault(m, set()).add(p)

    # ── Step 2: find root ─────────────────────────────────────────────
    # in_degree[A] = number of distinct B such that B => A
    in_degree: dict[str, int] = {f: 0 for f in feature_names}
    for src, targets in impl.items():
        for tgt in targets:
            if tgt in in_degree:
                in_degree[tgt] += 1

    root = None
    for f in feature_names:
        if f in core and in_degree[f] == 0:
            root = f
            break
    if root is None:
        # Fallback: transitive reachability — root is the feature reachable
        # from the most other features via implication chains.  Direct
        # in-degree can be misleading when intermediate edges are missing.
        reach_count: dict[str, int] = {}
        for f in feature_names:
            # Reverse BFS: find all features that can reach f
            visited: set[str] = set()
            queue = [f]
            while queue:
                node = queue.pop()
                if node in visited:
                    continue
                visited.add(node)
                for src, targets in impl.items():
                    if node in targets and src not in visited:
                        queue.append(src)
            reach_count[f] = len(visited) - 1  # exclude self
        root = max(feature_names, key=lambda f: reach_count[f])
    if root is None:
        root = feature_names[0]

    # ── Step 3: break 2-cycles from equivalences ───────────────────────
    # A == B injects both A→B and B→A.  Break the cycle BEFORE transitive
    # reduction so that TR doesn't remove real tree edges via equivalence
    # shortcuts (e.g. Fixed→Size removed because Fixed→Value→Size exists).
    # Heuristic chain (first that breaks the tie wins):
    #   0. More equivalence partners → structural hub (parent of eq partners).
    #   1. More outgoing non-eq edges → more "parent-like".
    #   2. More exclusion partners → part of an alternative/or group →
    #      structural tree node (parent); the eq partner with fewer
    #      exclusions is the leaf alias (child).
    #   3. Higher in-degree → closer to root → parent.
    direct_impl: dict[str, set[str]] = {a: set(bs) for a, bs in impl.items()}

    for pair in eq:
        a, b = tuple(pair)
        a_to_b = b in direct_impl.get(a, set())
        b_to_a = a in direct_impl.get(b, set())
        if a_to_b and b_to_a:
            # Tiebreak 0: equivalence partner count — a feature equivalent
            # to multiple others is a structural hub (mandatory parent).
            a_eq = sum(1 for p in eq if a in p)
            b_eq = sum(1 for p in eq if b in p)
            if a_eq != b_eq:
                if a_eq > b_eq:
                    direct_impl[a].discard(b)  # a is hub → parent
                else:
                    direct_impl[b].discard(a)
                continue
            a_out = len(direct_impl.get(a, set()) - {b})
            b_out = len(direct_impl.get(b, set()) - {a})
            if a_out != b_out:
                if a_out > b_out:
                    direct_impl[a].discard(b)
                else:
                    direct_impl[b].discard(a)
                continue
            # Tiebreak 1: exclusion participation count
            a_excl = sum(1 for p in excl if a in p)
            b_excl = sum(1 for p in excl if b in p)
            if a_excl != b_excl:
                if a_excl > b_excl:
                    direct_impl[a].discard(b)  # a is structural → parent
                else:
                    direct_impl[b].discard(a)
                continue
            # Tiebreak 2: in-degree (more incoming → closer to root → parent)
            if in_degree.get(a, 0) >= in_degree.get(b, 0):
                direct_impl[a].discard(b)
            else:
                direct_impl[b].discard(a)

    # Root can never be a child: remove any edge root → X (root implying
    # X means root is X's child, which is invalid).
    direct_impl.get(root, set()).clear()

    # ── Step 3b: transitive reduction of implication graph ────────────
    # Remove A→B if B is reachable from A via an alternate path of
    # length ≥ 2.  This properly handles multi-hop transitive chains
    # like Value → Fixed → Size → Stack.
    edges_to_remove: set[tuple[str, str]] = set()
    for a in list(direct_impl.keys()):
        for b in list(direct_impl.get(a, set())):
            # BFS from a's other neighbors to see if b is reachable
            visited: set[str] = set()
            queue = list(direct_impl.get(a, set()) - {b})
            found = False
            while queue and not found:
                node = queue.pop()
                if node == b:
                    found = True
                    break
                if node in visited:
                    continue
                visited.add(node)
                queue.extend(direct_impl.get(node, set()))
            if found:
                edges_to_remove.add((a, b))
    for a, b in edges_to_remove:
        direct_impl[a].discard(b)

    # ── Step 3c: redirect edges through equivalences ────────────────────
    # When A == B and B is now a mandatory child of A, the CA algorithm may
    # have learned X → A for features X that actually belong under B (since
    # A == B they are logically interchangeable).  Use completeness clauses
    # (B → or([X, ...])) to identify which features should be redirected
    # from X → A to X → B.
    for pair in eq:
        a, b = tuple(pair)
        a_to_b = b in direct_impl.get(a, set())
        b_to_a = a in direct_impl.get(b, set())
        # Determine parent/child after cycle-breaking
        if b_to_a and not a_to_b:
            parent_eq, child_eq = a, b
        elif a_to_b and not b_to_a:
            parent_eq, child_eq = b, a
        else:
            continue
        # Collect all features that child_eq claims via completeness clauses
        child_members: set[str] = set()
        for members in completeness.get(child_eq, []):
            child_members.update(members)
        for members in exactly_one.get(child_eq, []):
            child_members.update(members)
        # Also collect via parent_eq's completeness (since parent == child)
        for members in completeness.get(parent_eq, []):
            child_members.update(members)
        for members in exactly_one.get(parent_eq, []):
            child_members.update(members)
        if not child_members:
            # Fallback: use exclusion-based sibling detection.
            # Features that are children of child_eq and mutually exclusive
            # with features currently pointing to parent_eq should be
            # co-located under child_eq.
            child_children = {
                x
                for x, targets in direct_impl.items()
                if child_eq in targets and x != parent_eq
            }
            if child_children:
                for x, targets in list(direct_impl.items()):
                    if parent_eq in targets and x != child_eq:
                        # Does x share exclusions with any child_children?
                        has_excl = any(
                            frozenset({x, cc}) in excl for cc in child_children
                        )
                        if has_excl:
                            child_members.add(x)
        if not child_members:
            continue
        # Redirect: X → parent_eq becomes X → child_eq for claimed features
        for x in child_members:
            x_targets = direct_impl.get(x, set())
            if parent_eq in x_targets:
                x_targets.discard(parent_eq)
                x_targets.add(child_eq)

    # ── Step 3d: resolve multi-parent edges via constraint connectivity ─
    # After TR, a feature may still point to multiple potential parents
    # (e.g. Integer → Element_Type, Integer → Optimization, Integer →
    # Counter).  Only one can be the tree parent; the rest are cross-tree
    # "requires" constraints.
    #
    # Heuristic: for each candidate parent P, count how many of P's
    # other children (co-siblings of X) share ANY learned constraint
    # with X (implication, exclusion, or equivalence).  The parent whose
    # co-children form the strongest "clique" with X is the most likely
    # tree parent.  When the best score is > 0, keep only the winning
    # parent edge and discard the rest.
    all_linked: set[frozenset] = set()
    for a_name, targets in impl.items():
        for b_name in targets:
            all_linked.add(frozenset({a_name, b_name}))
    all_linked |= excl | eq

    edges_to_prune: list[tuple[str, str]] = []
    for x in direct_impl:
        targets = direct_impl.get(x, set())
        if len(targets) <= 1:
            continue
        scores: dict[str, int] = {}
        for p in targets:
            co_children = {
                other
                for other, other_targets in direct_impl.items()
                if other != x and p in other_targets
            }
            scores[p] = sum(
                1 for sib in co_children if frozenset({x, sib}) in all_linked
            )
        best_score = max(scores.values())
        if best_score > 0:
            best_parents = {p for p, s in scores.items() if s == best_score}
            for p in targets:
                if p not in best_parents:
                    edges_to_prune.append((x, p))
    for x, p in edges_to_prune:
        direct_impl.get(x, set()).discard(p)

    # ── Step 3e: break remaining cycles ──────────────────────────────
    # Despite equivalence cycle-breaking (step 3) and transitive reduction
    # (step 3b), cycles can persist from completeness-inferred edges
    # (step 1c).  Detect and break them by removing the back-edge with
    # the weakest support (fewest co-sibling links).
    def _find_cycle(graph: dict[str, set[str]], start: str) -> list[str] | None:
        """DFS from start; return cycle path if found."""
        visited, stack = set(), [(start, [start])]
        while stack:
            node, path = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            for nxt in graph.get(node, set()):
                if nxt == start:
                    return path + [nxt]
                if nxt not in visited:
                    stack.append((nxt, path + [nxt]))
        return None

    # Build reverse map: for edge A→P, parent_of[A] contains P
    # (A is child of P).  A cycle means P is also reachable from A
    # through other edges.
    changed = True
    max_iters = len(feature_names)
    while changed and max_iters > 0:
        changed = False
        max_iters -= 1
        for node in list(direct_impl.keys()):
            cycle = _find_cycle(direct_impl, node)
            if cycle and len(cycle) > 1:
                # Remove the weakest edge in the cycle
                weakest_edge = None
                weakest_score = float("inf")
                for ci in range(len(cycle) - 1):
                    a, b = cycle[ci], cycle[ci + 1]
                    # Score: number of constraints linking a to b's co-children
                    co_children = {
                        other
                        for other, other_targets in direct_impl.items()
                        if other != a and b in other_targets
                    }
                    score = sum(
                        1 for sib in co_children if frozenset({a, sib}) in all_linked
                    )
                    if score < weakest_score:
                        weakest_score = score
                        weakest_edge = (a, b)
                if weakest_edge:
                    direct_impl.get(weakest_edge[0], set()).discard(weakest_edge[1])
                    changed = True
                    break  # restart after each removal

    # ── Step 3f: resolve remaining multi-parent edges ────────────────
    # After all the above steps, a feature may still point to multiple
    # parents.  Keep only the best one.
    for x in list(direct_impl.keys()):
        targets = direct_impl.get(x, set())
        if len(targets) <= 1:
            continue
        # Score each candidate parent by co-sibling connectivity
        scores: dict[str, int] = {}
        for p in targets:
            co_children = {
                other
                for other, other_targets in direct_impl.items()
                if other != x and p in other_targets
            }
            scores[p] = sum(
                1 for sib in co_children if frozenset({x, sib}) in all_linked
            )
        best_score = max(scores.values())
        best_parents = {p for p, s in scores.items() if s == best_score}
        # If tied, prefer the parent with more total children (more populated subtree)
        if len(best_parents) > 1:
            child_counts = {}
            for p in best_parents:
                child_counts[p] = sum(
                    1
                    for other, other_targets in direct_impl.items()
                    if p in other_targets
                )
            best_count = max(child_counts.values())
            best_parents = {p for p, c in child_counts.items() if c == best_count}
        keep = next(iter(best_parents))
        for p in list(targets):
            if p != keep:
                targets.discard(p)

    # ── Step 4: group children by parent ──────────────────────────────
    # Edge A→P means A is a child of P.
    children_of: dict[str, set[str]] = {f: set() for f in feature_names}
    for a, targets in direct_impl.items():
        for p in targets:
            if p in children_of:
                children_of[p].add(a)

    # ── Step 5: classify and sub-group children ─────────────────────
    # Instead of lumping all children into one group, use completeness
    # and exactly-one hints to identify multi-child groups (or/alternative)
    # and make remaining children individual optional/mandatory groups.
    tree_info: dict = {}
    attached: set[str] = set()

    for parent in feature_names:
        children = list(children_of[parent])
        if not children:
            continue
        attached.update(children)
        children_set = set(children)
        used: set[str] = set()
        groups: list[tuple[str, list[str]]] = []

        # 5a. Exactly-one hints → alternative groups
        for hint_set in exactly_one.get(parent, []):
            matched = (children_set & hint_set) - used
            if len(matched) >= 2:
                groups.append(("alternative", sorted(matched)))
                used.update(matched)

        # 5b. Completeness hints → or / alternative groups
        # Intersect all hints (filtered to actual unused children) to find
        # the core group members that appear in every hint.
        parent_comp = completeness.get(parent, [])
        if parent_comp and (children_set - used):
            filtered = []
            for hint_set in parent_comp:
                subset = (hint_set & children_set) - used
                if len(subset) >= 2:
                    filtered.append(subset)
            if filtered:
                core = filtered[0]
                for h in filtered[1:]:
                    core = core & h
                if len(core) >= 2:
                    core_list = sorted(core)
                    all_excl = all(
                        frozenset({core_list[i], core_list[j]}) in excl
                        for i in range(len(core_list))
                        for j in range(i + 1, len(core_list))
                    )
                    gtype = "alternative" if all_excl else "or"
                    groups.append((gtype, core_list))
                    used.update(core)

        # 5c. Remaining children: check if all mutually exclusive → alternative
        remaining = [c for c in children if c not in used]
        if len(remaining) >= 2:
            excl_children = []
            for i in range(len(remaining)):
                for j in range(i + 1, len(remaining)):
                    cs = frozenset({remaining[i], remaining[j]})

                    if cs in excl:
                        excl_children.append(cs)

            alternative_sets = merge_shared_sets(excl_children)

            for alt_set in alternative_sets:
                groups.append(("alternative", sorted(alt_set)))
                used.update(alt_set)
                remaining = [c for c in remaining if c not in alt_set]

        # 5d. Individual children: mandatory (if equivalence) or optional
        for c in remaining:
            if frozenset({parent, c}) in eq:
                groups.append(("mandatory", [c]))
            else:
                groups.append(("optional", [c]))

        tree_info[parent] = groups

    # ── Step 6: attach unattached features ────────────────────────────
    unattached = [f for f in feature_names if f != root and f not in attached]
    if unattached:
        tree_info.setdefault(root, []).append(("optional", unattached))

    return tree_info


# ── Tree-guided group refinement (post-inference) ─────────────────────


def refine_tree_groups(
    feature_names: list[str],
    variables: list,
    learned_cl: list,
    tree_info: dict,
    oracle: ConstraintOracle,
) -> tuple[list, dict, int]:
    """Refine inferred tree group types using targeted oracle queries.

    After binary CA converges and the tree has been inferred from learned
    binary constraints, this function asks a small number of oracle queries
    to confirm or correct group classifications that binary constraints
    alone cannot determine:

    1. **Root core**: Is the root always selected?
    2. **Completeness**: For parents with ≥2 individual optional children,
       does the parent require at least one?  (optionals → or/alternative)
    3. **Alternative vs or**: For or-groups or newly-merged groups, can
       multiple children be selected simultaneously?
    4. **Mandatory**: For single optional children, is the child required
       when the parent is selected?

    Worst-case query count: O(parents + leaves), typically much less.

    Returns ``(enhanced_cl, updated_tree_info, n_queries)``.
    """
    var_of = {name: v for name, v in zip(feature_names, variables)}
    added_cl: list = []
    n_queries = 0

    # Deep copy tree_info (use lists internally for mutability)
    updated_tree: dict[str, list] = {}
    for parent, groups in tree_info.items():
        updated_tree[parent] = [[gt, list(ch)] for gt, ch in groups]

    # Find root (feature that never appears as a child)
    all_children: set[str] = set()
    for groups in tree_info.values():
        for _, children in groups:
            all_children.update(children)
    root = next(
        (f for f in feature_names if f not in all_children),
        feature_names[0],
    )

    # Pre-compute structural constraints from groups that infer_tree
    # already classified (from binary exclusion/implication patterns).
    # These ensure the solver creates assignments that respect known
    # group structure — preventing mis-attribution when testing other
    # groups.
    structural_cl: list = []
    for parent, groups in tree_info.items():
        pv = var_of[parent]
        for gtype, children in groups:
            cvs = [var_of[c] for c in children]
            if gtype in ("or", "alternative"):
                structural_cl.append(pv.implies(cp.any(cvs)))
                if gtype == "alternative":
                    for i in range(len(cvs)):
                        for j in range(i + 1, len(cvs)):
                            structural_cl.append(cvs[i].implies(~cvs[j]))
            elif gtype == "mandatory" and len(children) == 1:
                structural_cl.append(pv.implies(cvs[0]))

    # Build ancestor map from tree_info for reachability forcing
    tree_edges: set[tuple[str, str]] = set()  # (child, parent)
    for parent, groups in tree_info.items():
        for _, children in groups:
            for child in children:
                tree_edges.add((child, parent))

    ancestors: dict[str, set[str]] = {f: set() for f in feature_names}
    for child, parent in tree_edges:
        ancestors[child].add(parent)
    changed = True
    while changed:
        changed = False
        for child in feature_names:
            for parent in list(ancestors[child]):
                new_anc = ancestors[parent] - ancestors[child]
                if new_anc:
                    ancestors[child].update(new_anc)
                    changed = True

    def _ask(
        assumptions: list,
        *,
        exclude_excl: set[frozenset] | None = None,
        retries: int = 0,
    ) -> bool | None:
        """Find SAT assignment under assumptions and ask the oracle.

        After solve(), the cpmpy variables hold the solver's values.
        Passing the variable objects to the oracle lets it evaluate
        constraints using those values — no target access needed.

        If *exclude_excl* is given, learned constraints that encode
        mutual exclusion between any pair in the set are dropped from
        the solver.  This is critical for alternative-vs-or testing:
        CA may have learned spurious exclusion constraints between
        group siblings, making the pair test UNSAT when the oracle
        would actually accept them coexisting.

        If *retries* > 0 and the oracle rejects the assignment, the
        solver is re-invoked with a constraint excluding the current
        assignment, up to *retries* additional attempts.  This helps
        find valid configurations when the solver's initial assignment
        violates unrelated ground-truth constraints.

        Returns True (oracle accepts), False (oracle rejects), or
        None (assumptions + learned are UNSAT — property already implied).
        """
        nonlocal n_queries
        if exclude_excl:
            base_cl = [c for c in learned_cl if not _is_excl_between(c, exclude_excl)]
            base_cl += [
                c for c in structural_cl if not _is_excl_between(c, exclude_excl)
            ]
        else:
            base_cl = list(learned_cl) + list(structural_cl)
        base_cl = base_cl + added_cl

        feats = []
        for c in assumptions:
            feats.extend(_mentioned_features(c))
        # Reachability: all ancestors of mentioned features must be True
        reach = set()
        for f in feats:
            reach.update(ancestors.get(f, set()))
        reachability = [var_of[a] for a in reach if a in var_of]

        # TODO If reachability works, we might not need blocking anymore
        # TODO Need all mandatory requirements for all parents in reachability, too
        blocking: list = []
        for attempt in range(1 + retries):
            m = cp.Model(base_cl + assumptions + reachability + blocking)
            if not m.solve():
                return None if attempt == 0 else False
            n_queries += 1
            try:
                result = oracle.answer_membership_query(list(variables))
            except Exception as e:
                if "Collapse" in str(e):
                    return None
                raise
            if result is True or attempt >= retries:
                return result
            # Oracle rejected — block this assignment and retry
            assignment = [v if v.value() else ~v for v in variables]
            blocking.append(~cp.all(assignment))

        return False

    def _is_excl_between(c, pairs: set[frozenset]) -> bool:
        """Check if constraint c encodes A => ~B for a pair in the set."""
        if not (isinstance(c, Operator) and c.name == "->"):
            return False
        lhs, rhs = c.args
        a = (
            lhs.name
            if isinstance(lhs, _BoolVarImpl) and not isinstance(lhs, NegBoolView)
            else None
        )
        if a is None:
            return False
        if isinstance(rhs, NegBoolView):
            b = rhs._bv.name
            return frozenset({a, b}) in pairs
        return False

    # ── 1. Root core check ────────────────────────────────────────
    root_var = var_of[root]
    ans = _ask([~root_var])
    if ans is None or ans is False:
        added_cl.append(root_var)
        print(f"  refine_tree: root '{root}' is core")

    # ── 2. Process each parent's groups (bottom-up) ─────────────
    # Process deepest nodes first so that inner group constraints
    # (e.g. Element_Type → or(Int, Float, String)) are established
    # before testing outer groups.  This prevents mis-attribution
    # when the solver picks assignments that violate unrelated groups.
    depths: dict[str, int] = {}
    queue_bfs = [(root, 0)]
    while queue_bfs:
        node, d = queue_bfs.pop(0)
        if node in depths:
            continue
        depths[node] = d
        for _, children in updated_tree.get(node, []):
            for child in children:
                if child not in depths:
                    queue_bfs.append((child, d + 1))
    processing_order = sorted(feature_names, key=lambda f: -depths.get(f, 0))

    for parent in processing_order:
        if parent not in updated_tree:
            continue
        groups = updated_tree[parent]
        pv = var_of[parent]

        # 2a. Merge individual optionals → test completeness
        ind_idx = [
            i for i, (gt, ch) in enumerate(groups) if gt == "optional" and len(ch) == 1
        ]
        if len(ind_idx) >= 2:
            ind_children = [groups[i][1][0] for i in ind_idx]
            cvs = [var_of[c] for c in ind_children]

            # Completeness: parent=T, all these children=F
            assumption = [pv] + [~cv for cv in cvs]
            ans = _ask(assumption)
            is_complete = ans is None or ans is False

            if is_complete:
                # Alternative vs or: test multiple pairs to determine
                # if any two children can coexist.  Exclude learned
                # exclusion constraints between group children so that
                # spurious CA exclusions don't force UNSAT.
                # AT LEAST ONE OF THEM MUST BE TRUE
                child_pairs = {
                    frozenset({ind_children[i], ind_children[j]})
                    for i in range(len(ind_children))
                    for j in range(i + 1, len(ind_children))
                }
                is_alt = True
                for pi in range(len(cvs)):
                    for pj in range(pi + 1, len(cvs)):
                        pair_ans = _ask(
                            [pv, cvs[pi], cvs[pj]],
                            exclude_excl=child_pairs,
                            retries=3,
                        )
                        if pair_ans is True:
                            is_alt = False
                            break
                    if not is_alt:
                        break

                gtype = "alternative" if is_alt else "or"
                added_cl.append(pv.implies(cp.any(cvs)))
                if is_alt:
                    for i in range(len(cvs)):
                        for j in range(i + 1, len(cvs)):
                            added_cl.append(cvs[i].implies(~cvs[j]))

                # Replace individual optionals with merged group
                keep = [g for i, g in enumerate(groups) if i not in set(ind_idx)]
                keep.append([gtype, sorted(ind_children)])
                updated_tree[parent] = keep
                groups = keep
                print(f"  refine_tree: {parent} → {gtype} {ind_children}")
            else:
                # ALL CAN BE FALSE -> optional
                pass

        # 2b-pre. Existing alternative/or groups → verify completeness & type
        # First check completeness: can parent exist without any child?
        # If yes, decompose multi-child group into individual optionals.
        # Then check alternative vs or.
        new_groups = list(groups)
        decomposed = False
        for i, (gt, ch) in enumerate(groups):
            if gt in ("alternative", "or") and len(ch) >= 2:
                cvs = [var_of[c] for c in ch]
                # Completeness check: parent on, all group children off
                comp_ans = _ask([pv] + [~cv for cv in cvs])
                if comp_ans is True:
                    # Not complete: decompose into individual optionals
                    new_groups[i] = None  # mark for removal
                    for c in ch:
                        new_groups.append(["optional", [c]])
                    decomposed = True
                    print(f"  refine_tree: {parent} {gt} {ch} → optionals (not complete)")
                    continue
                # Check alternative vs or: can any two coexist?
                # Exclude learned exclusion constraints between group
                # children so spurious CA exclusions don't block the test.
                child_pairs = {
                    frozenset({ch[ci], ch[cj]})
                    for ci in range(len(ch))
                    for cj in range(ci + 1, len(ch))
                }
                found_coexist = False
                for pi in range(len(cvs)):
                    for pj in range(pi + 1, len(cvs)):
                        ans = _ask(
                            [pv, cvs[pi], cvs[pj]],
                            exclude_excl=child_pairs,
                            retries=3,
                        )
                        if ans is True:
                            found_coexist = True
                            break
                    if found_coexist:
                        break
                if found_coexist and gt == "alternative":
                    groups[i] = ["or", ch]
                    print(f"  refine_tree: {parent} {ch} alternative → or")
                elif not found_coexist and gt == "or":
                    groups[i] = ["alternative", ch]
                    for ci in range(len(cvs)):
                        for cj in range(ci + 1, len(cvs)):
                            added_cl.append(cvs[ci].implies(~cvs[cj]))
                    print(f"  refine_tree: {parent} {ch} or → alternative")
        if decomposed:
            updated_tree[parent] = [g for g in new_groups if g is not None]
            groups = updated_tree[parent]

        # 2c. Single optional → maybe mandatory
        for i, (gt, ch) in enumerate(groups):
            if gt == "optional" and len(ch) == 1:
                cv = var_of[ch[0]]
                ans = _ask([pv, ~cv])
                if ans is None or ans is False:
                    groups[i] = ["mandatory", ch]
                    added_cl.append(pv.implies(cv))
                    print(f"  refine_tree: {parent}/{ch[0]} → mandatory")

    # ── 3. Verify parent-child relationships ─────────────────────────
    # For each child, check that child → parent is a valid constraint
    # (i.e., the child cannot be active without the parent).  If the
    # oracle accepts child=T, parent=F, this is a cross-tree constraint
    # misread as a tree edge — remove the child from this parent.
    for parent in list(updated_tree.keys()):
        if parent == root:
            continue
        pv = var_of[parent]
        new_groups = []
        relocated: list[str] = []
        for gt, ch in updated_tree[parent]:
            valid_children = []
            for c in ch:
                cv = var_of[c]
                ans = _ask([cv, ~pv])
                if ans is True:
                    # Child can exist without parent → not a tree child
                    relocated.append(c)
                    print(f"  refine_tree: {c} not child of {parent} (relocating to root)")
                else:
                    valid_children.append(c)
            if valid_children:
                new_groups.append([gt, valid_children])
        if relocated:
            updated_tree[parent] = new_groups if new_groups else []
            if not updated_tree[parent]:
                del updated_tree[parent]
            # Add relocated children as optionals under root
            root_groups = updated_tree.setdefault(root, [])
            for c in relocated:
                root_groups.append(["optional", [c]])

    # Convert back to tuple format
    final_tree = {p: [(g[0], g[1]) for g in gs] for p, gs in updated_tree.items()}

    enhanced_cl = list(learned_cl) + added_cl
    print(f"  refine_tree: +{len(added_cl)} constraint(s), {n_queries} oracle queries")
    return enhanced_cl, final_tree, n_queries


# ── Tree validation ────────────────────────────────────────────────────


def _validate_tree(feature_names: list[str], tree_info: dict) -> list[str]:
    """Validate that the inferred tree is a valid UVL tree structure.

    Checks:
    1. Single root: exactly one feature should never appear as a child
    2. Single parent invariant: each feature should appear as a child under at most one parent
    3. No cycles: tree must be acyclic
    4. Reachability: all features must be reachable from root

    Returns list of error messages (empty if valid).
    """
    errors = []

    # Collect: parents_of[child] = set of parents for that child
    parents_of: dict[str, set[str]] = {f: set() for f in feature_names}
    children_of_parent: dict[str, set[str]] = {f: set() for f in feature_names}

    for parent, groups in tree_info.items():
        for gtype, children in groups:
            for child in children:
                parents_of[child].add(parent)
                children_of_parent[parent].add(child)

    # Check 1: Single parent invariant
    multi_parent_features = []
    for f in feature_names:
        num_parents = len(parents_of[f])
        if num_parents > 1:
            multi_parent_features.append((f, num_parents, parents_of[f]))

    if multi_parent_features:
        errors.append(
            f"Multiple parents violation: "
            f"{len(multi_parent_features)} features appear under multiple parents:\n"
            + "\n".join(
                f"  - {name}: parents = {[p for p in parents]}"
                for name, num_parents, parents in multi_parent_features
            )
        )

    # Check 2: No cycles
    # Build adjacency list (parent -> children) for tree traversal
    adj: dict[str, set[str]] = {f: set() for f in feature_names}
    for parent, children in children_of_parent.items():
        adj[parent] = children

    # DFS-based cycle detection
    visited = set()
    rec_stack = set()
    cycles = []

    def dfs(node, path):
        visited.add(node)
        rec_stack.add(node)
        path.append(node)

        for neighbor in adj.get(node, []):
            if neighbor not in visited:
                dfs(neighbor, path)
            elif neighbor in rec_stack:
                # Found cycle
                cycle_start = path.index(neighbor)
                cycle = path[cycle_start:] + [neighbor]
                cycles.append(cycle)

        path.pop()
        rec_stack.remove(node)

    for f in feature_names:
        if f not in visited:
            dfs(f, [])

    if cycles:
        errors.append(
            f"Cycles detected in tree structure ({len(cycles)} cycle(s)):\n"
            + "\n".join(f"  {' -> '.join(c)}" for c in cycles)
        )

    # Check 3: All features should be reachable from root
    # Find root (feature with no parents)
    roots = [f for f in feature_names if len(parents_of[f]) == 0]

    if len(roots) != 1:
        if len(roots) == 0:
            errors.append("No root found: all features have at least one parent")
        else:
            errors.append(
                f"Multiple roots found ({len(roots)}): "
                f"{', '.join(roots)}. Expected exactly one root."
            )

    # Check reachability from root
    if roots:
        root = roots[0]
        # BFS to find all reachable nodes
        visited_nodes = set()
        queue = [root]
        while queue:
            node = queue.pop(0)
            if node in visited_nodes:
                continue
            visited_nodes.add(node)
            for child in adj.get(node, []):
                if child not in visited_nodes:
                    queue.append(child)

        unreachable = [f for f in feature_names if f not in visited_nodes]
        if unreachable:
            errors.append(
                f"{len(unreachable)} feature(s) not reachable from root: "
                f"{', '.join(unreachable)}"
            )

    return errors


def _fix_multi_parent_tree(
    feature_names: list[str], tree_info: dict
) -> tuple[dict, list[str]]:
    """Fix multiple parent violations using the heuristic from README.

    When a feature has multiple inferred parents, keep only the parent with
    the most children already assigned (most "populated" subtree), and demote
    the others to cross-tree constraints (recorded in returned cross_tree_cl).

    Returns:
        (fixed_tree_info, list_of_demoted_constraints_as_strings)
    """
    fixed_tree_info = {k: [list(g) for g in v] for k, v in tree_info.items()}
    cross_tree_cl = []

    # Build maps: parents_of[child] = set of parents, children_count[parent] = total children
    parents_of: dict[str, set[str]] = {f: set() for f in feature_names}
    children_count: dict[str, int] = {f: 0 for f in feature_names}

    for parent, groups in fixed_tree_info.items():
        for gtype, children in groups:
            for child in children:
                parents_of[child].add(parent)
            children_count[parent] += len(children)

    # Identify features with multiple parents
    for f in feature_names:
        if len(parents_of[f]) <= 1:
            continue
        parents = list(parents_of[f])

        # Keep the deeper (more specific) parent, not the root catch-all.
        # Heuristic: prefer the parent that is itself a child of another
        # feature (i.e., NOT the root), and among non-root parents prefer
        # the one with fewer children (more specific/smaller group).
        root_names = {f for f in feature_names if not parents_of[f]}

        def _parent_score(p):
            # Non-root parents get priority (score 0), root gets score 1
            is_root = 1 if p in root_names else 0
            return (is_root, children_count[p])

        parents.sort(key=_parent_score)
        best_parent = parents[0]
        demoted_parents = parents[1:]

        # Remove f from all demoted parents' groups
        for p in demoted_parents:
            if p not in fixed_tree_info:
                continue
            new_groups = []
            for gtype, children in fixed_tree_info[p]:
                if f in children:
                    children = [c for c in children if c != f]
                if children:
                    new_groups.append([gtype, children])
            if new_groups:
                fixed_tree_info[p] = new_groups
            else:
                del fixed_tree_info[p]
            cross_tree_cl.append(f"{f} => {p}")

    # Convert back to tuple format
    return {
        k: [(g[0], g[1]) for g in v] for k, v in fixed_tree_info.items()
    }, cross_tree_cl


# ── Constraint reconstruction from inferred tree ──────────────────


def constraints_from_tree(
    feature_names: list[str],
    variables: list,
    tree_info: dict,
    learned_cl: list,
) -> list:
    """Reconstruct a CPMpy constraint model from the inferred tree structure.

    Generates structural constraints directly from ``tree_info`` (child→parent
    edges, group-completeness for or/alternative groups, mutual exclusions for
    alternative groups, mandatory equivalences) and then appends any constraints
    from ``learned_cl`` that are *not* already encoded by the tree (cross-tree
    requires/excludes, equivalences not captured as mandatory pairs, unary
    core/dead).

    The result can be passed to ``verify_learned`` to test whether the
    tree reconstruction — together with residual learned constraints —
    is semantically equivalent to the oracle target.

    Notes
    -----
    For alternative groups the tree encoding adds *both* directions of each
    mutual exclusion (``Ci => ~Cj`` and ``Cj => ~Ci``) even if only one was
    learned; this is fine because both directions are the same SAT clause.
    """
    var_of = {name: v for name, v in zip(feature_names, variables)}

    # ── 1. Structural constraints from tree ────────────────────────────
    structural_cl: list = []
    tree_edges: set[tuple[str, str]] = set()  # (child, parent) direct edges
    alt_pairs: set[frozenset] = set()  # {c1, c2} alternative siblings
    mandatory_pairs: set[frozenset] = set()  # {parent, child} mandatory groups
    or_alt_parents: set[str] = set()  # parents with completeness clauses

    # Root feature is always selected in a feature model
    all_children: set[str] = set()
    for groups in tree_info.values():
        for _, children in groups:
            all_children.update(children)
    for name in feature_names:
        if name not in all_children:
            structural_cl.append(var_of[name])
            break

    for parent, groups in tree_info.items():
        pv = var_of[parent]
        for gtype, children in groups:
            cvs = [var_of[c] for c in children]
            # child → parent holds for every group type
            for c, cv in zip(children, cvs):
                tree_edges.add((c, parent))
                structural_cl.append(cv.implies(pv))

            if gtype == "mandatory":
                # parent ↔ child (single mandatory child)
                structural_cl.append(pv.implies(cvs[0]))
                mandatory_pairs.add(frozenset({parent, children[0]}))
            elif gtype in ("or", "alternative"):
                # group completeness: parent → at least one child
                structural_cl.append(pv.implies(cp.any(cvs)))
                or_alt_parents.add(parent)
                if gtype == "alternative":
                    # mutual exclusion between every pair of children
                    for i in range(len(cvs)):
                        for j in range(i + 1, len(cvs)):
                            structural_cl.append(cvs[i].implies(~cvs[j]))
                            alt_pairs.add(frozenset({children[i], children[j]}))
            # "optional": only child → parent (already added above)

    # ── 2. Cross-tree constraints from learned ─────────────────────────
    # Retain learned constraints not already encoded by the tree structure.
    # Compute tree-reachable ancestors for each feature so we can skip
    # any A => B where B is an ancestor of A in the tree.
    tree_ancestors: dict[str, set[str]] = {f: set() for f in feature_names}
    for child, parent in tree_edges:
        tree_ancestors[child].add(parent)
    # Propagate: ancestors of my parent are also my ancestors
    changed = True
    while changed:
        changed = False
        for child in feature_names:
            for parent in list(tree_ancestors[child]):
                new_anc = tree_ancestors[parent] - tree_ancestors[child]
                if new_anc:
                    tree_ancestors[child].update(new_anc)
                    changed = True

    cross_tree_cl: list = []
    for c in learned_cl:
        if isinstance(c, Operator) and c.name == "->":
            lhs, rhs = c.args
            if isinstance(lhs, _BoolVarImpl) and not isinstance(lhs, NegBoolView):
                if isinstance(rhs, _BoolVarImpl) and not isinstance(rhs, NegBoolView):
                    # A => B: skip if B is a tree ancestor of A (direct or transitive)
                    if rhs.name in tree_ancestors.get(lhs.name, set()):
                        continue
                elif isinstance(rhs, NegBoolView):
                    # A => ~B: skip if it's an alternative-sibling exclusion
                    if frozenset({lhs.name, rhs._bv.name}) in alt_pairs:
                        continue
                else:
                    # A => any([...]): skip if A is an or/alt parent (already have completeness)
                    if lhs.name in or_alt_parents:
                        continue
            cross_tree_cl.append(c)
        elif isinstance(c, Comparison) and c.name == "==":
            a_expr, b_expr = c.args
            if (
                isinstance(a_expr, _BoolVarImpl)
                and not isinstance(a_expr, NegBoolView)
                and isinstance(b_expr, _BoolVarImpl)
                and not isinstance(b_expr, NegBoolView)
            ):
                # A == B: skip if it's encoded as a mandatory pair
                if frozenset({a_expr.name, b_expr.name}) in mandatory_pairs:
                    continue
            cross_tree_cl.append(c)
        else:
            # Unary constraints (core/dead): include if not already part of the tree
            if c not in structural_cl:
                cross_tree_cl.append(c)

    return structural_cl + cross_tree_cl, cross_tree_cl


# ── Constraint cleanup (post-tree, pre-verification) ────────────────


def cleanup_dumb(
    feature_names: list[str],
    variables: list,
    learned_cl: list,
    tree_info: dict,
    oracle: ConstraintOracle,
):
    def _negate(c):
        negation = [~c]

        if isinstance(c, Operator) and c.name == "->":
            lhs, rhs = c.args
            if isinstance(lhs, _BoolVarImpl) and not isinstance(lhs, NegBoolView):
                # Force the antecedent True
                if isinstance(rhs, _BoolVarImpl) and not isinstance(rhs, NegBoolView):
                    # A => B: force A=T, B=F
                    negation = [lhs, ~rhs]
                elif isinstance(rhs, NegBoolView):
                    # A => ~B: force A=T, B=T
                    negation = [lhs, rhs._bv]
                else:
                    # A => complex_rhs: force A=T, negate rhs
                    negation = [lhs, ~rhs]

        elif isinstance(c, Comparison) and c.name == "==":
            a_expr, b_expr = c.args
            if (
                isinstance(a_expr, _BoolVarImpl)
                and not isinstance(a_expr, NegBoolView)
                and isinstance(b_expr, _BoolVarImpl)
                and not isinstance(b_expr, NegBoolView)
            ):
                # A == B: try A=T, B=F
                # negation = [a_expr, ~b_expr]
                negation = [sum([a_expr, b_expr]) == 1]

        return negation

    removed: list = []
    n_queries = 0
    remaining = list(learned_cl)
    changed = True

    while changed:
        changed = False
        kept_cross: list = []

        for i, c in enumerate(remaining):
            negation = _negate(c)
            all_cl = [c2 for c2 in learned_cl if c2 not in [c] and c2 not in removed]
            # TODO Not sure we actually need reachability
            m = cp.Model(all_cl + negation)
            if not m.solve():
                # Can't violate c while satisfying everything else → redundant, keep
                kept_cross.append(c)
                print(f"  cleanup: KEEP (implied) {c}")
                continue

            # Ask the oracle
            n_queries += 1
            try:
                # TODO Some form of caching of queries would be nice,
                # in this loop it can happen that we query the same thing multiple times
                oracle_accepts = oracle.answer_membership_query(list(variables))
            except Exception as e:
                if "Collapse" in str(e):
                    kept_cross.append(c)
                    print(f"  cleanup: KEEP (collapse) {c}")
                    continue
                raise

            if oracle_accepts:
                # Oracle accepts the c-violating assignment → c is spurious
                removed.append(c)
                print(f"  cleanup: REMOVE {c}")
                changed = True
            else:
                kept_cross.append(c)
                print(f"  cleanup: KEEP (oracle rejected) {c}")

        remaining = kept_cross

    print(f"  cleanup: removed {len(removed)} / {len(learned_cl)} constraint(s) in {n_queries} queries")
    # cleaned_cl = list(learned_cl)  + kept_cross
    cleaned_cl = [c for c in learned_cl if c not in removed]
    return cleaned_cl, removed, n_queries


def cleanup_constraints(
    feature_names: list[str],
    variables: list,
    learned_cl: list,
    tree_info: dict,
    oracle: ConstraintOracle,
) -> tuple[list, list, int]:
    """Remove spurious cross-tree constraints by generating per-constraint counter-examples.

    For each cross-tree constraint C (i.e. constraints not implied by the tree
    structure), we try to find an assignment that:

    1. Satisfies all tree-structural constraints (child→parent, group
       completeness, alternative exclusions, mandatory links).
    2. Satisfies all other cross-tree constraints (everything except C).
    3. **Violates** C.
    4. Keeps the features mentioned in C **reachable** — their ancestor path
       in the tree is forced active so the test is non-vacuous.

    If such an assignment exists and the oracle **accepts** it, then C is
    spurious (the ground truth does not require it) and we remove it.

    If the solver says UNSAT, C is implied by the remaining constraints and
    is therefore redundant but not wrong — we keep it.

    If the oracle rejects the assignment, C is genuinely needed — we keep it.

    Parameters
    ----------
    feature_names : list[str]
    variables : list of CPMpy BoolVar
    learned_cl : list of CPMpy constraints (flat CA + completeness + tree-refined)
    tree_info : dict  {parent: [(gtype, [children]), ...]}
    oracle : ConstraintOracle

    Returns
    -------
    (cleaned_cl, removed_cl, n_queries)
    """
    var_of = {name: v for name, v in zip(feature_names, variables)}

    # ── Build tree-structural constraints and classify learned ────────
    structural_cl, cross_tree_cl = constraints_from_tree(
        feature_names, variables, tree_info, learned_cl
    )

    # Build ancestor map from tree_info for reachability forcing
    tree_edges: set[tuple[str, str]] = set()  # (child, parent)
    for parent, groups in tree_info.items():
        for _, children in groups:
            for child in children:
                tree_edges.add((child, parent))

    ancestors: dict[str, set[str]] = {f: set() for f in feature_names}
    for child, parent in tree_edges:
        ancestors[child].add(parent)
    changed = True
    while changed:
        changed = False
        for child in feature_names:
            for parent in list(ancestors[child]):
                new_anc = ancestors[parent] - ancestors[child]
                if new_anc:
                    ancestors[child].update(new_anc)
                    changed = True

    # ── Helper: negate a constraint ───────────────────────────────────
    def _negate_with_reachability(c) -> tuple[list, list]:
        """Return (negation_assumptions, reachability_assumptions).

        negation_assumptions: list of CPMpy expressions that together negate C
            while keeping the relevant features active.
        reachability_assumptions: list of BoolVars forced True (ancestors).
        """
        feats = _mentioned_features(c)
        # Reachability: all ancestors of mentioned features must be True
        reach = set()
        for f in feats:
            reach.update(ancestors.get(f, set()))
        reachability = [var_of[a] for a in reach if a in var_of]

        # Also force the features themselves to be active where sensible.
        # For A => B: force A=True (makes the implication non-vacuous),
        #   the negation then requires B=False.
        # For A => ~B: force A=True, negation requires B=True.
        # For A == B: try A=True, B=False first.
        # Generic fallback: just negate the whole constraint.
        negation = [~c]

        if isinstance(c, Operator) and c.name == "->":
            lhs, rhs = c.args
            if isinstance(lhs, _BoolVarImpl) and not isinstance(lhs, NegBoolView):
                # Force the antecedent True
                if isinstance(rhs, _BoolVarImpl) and not isinstance(rhs, NegBoolView):
                    # A => B: force A=T, B=F
                    negation = [lhs, ~rhs]
                elif isinstance(rhs, NegBoolView):
                    # A => ~B: force A=T, B=T
                    negation = [lhs, rhs._bv]
                else:
                    # A => complex_rhs: force A=T, negate rhs
                    negation = [lhs, ~rhs]

        elif isinstance(c, Comparison) and c.name == "==":
            a_expr, b_expr = c.args
            if (
                isinstance(a_expr, _BoolVarImpl)
                and not isinstance(a_expr, NegBoolView)
                and isinstance(b_expr, _BoolVarImpl)
                and not isinstance(b_expr, NegBoolView)
            ):
                # A == B: try A=T, B=F
                negation = [a_expr, ~b_expr]

        return negation, reachability

    # ── Main loop: test each cross-tree constraint ────────────────────
    removed: list = []
    n_queries = 0
    remaining = list(cross_tree_cl)
    changed = True

    while changed:
        changed = False
        kept_cross: list = []

        for i, c in enumerate(remaining):
            negation, reachability = _negate_with_reachability(c)

            # Build the constraint set: structural + already-kept cross-tree + all
            # remaining untested cross-tree (conservative: don't remove a constraint
            # that's only needed because another spurious one was kept).
            other_cross = kept_cross + remaining[i + 1 :]
            all_cl = list(structural_cl) + other_cross
            all_cl = [c2 for c2 in all_cl if c2 not in [c] and c2 not in removed]
            # TODO Not sure we actually need reachability
            m = cp.Model(all_cl + reachability + negation)
            if not m.solve():
                # Can't violate c while satisfying everything else → redundant, keep
                kept_cross.append(c)
                print(f"  cleanup: KEEP (implied) {c}")
                continue

            # Ask the oracle
            n_queries += 1
            try:
                # TODO Some form of caching of queries would be nice,
                # in this loop it can happen that we query the same thing multiple times
                oracle_accepts = oracle.answer_membership_query(list(variables))
            except Exception as e:
                if "Collapse" in str(e):
                    kept_cross.append(c)
                    print(f"  cleanup: KEEP (collapse) {c}")
                    continue
                raise

            if oracle_accepts:
                # Oracle accepts the c-violating assignment → c is spurious
                removed.append(c)
                print(f"  cleanup: REMOVE {c}")
                changed = True
            else:
                kept_cross.append(c)
                print(f"  cleanup: KEEP (oracle rejected) {c}")

        remaining = kept_cross

    print(f"  cleanup: removed {len(removed)} / {len(cross_tree_cl)} cross-tree constraint(s) in {n_queries} queries")

    # Return: structural + surviving cross-tree
    cleaned_cl = list(structural_cl) + kept_cross
    cleaned_cl = [c for c in cleaned_cl if c not in removed]
    return cleaned_cl, removed, n_queries
