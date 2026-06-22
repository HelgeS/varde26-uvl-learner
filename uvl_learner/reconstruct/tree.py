"""Tree inference, validation, and single-parent repair.

- infer_tree() — reconstruct a parent→groups tree from learned constraints.
- _validate_tree() / _fix_multi_parent_tree() — enforce a single-parent tree.
"""

import cpmpy as cp
from cpmpy.expressions.variables import _BoolVarImpl, NegBoolView
from cpmpy.expressions.core import Operator, Comparison
from pycona import ConstraintOracle

from ._common import merge_shared_sets


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


