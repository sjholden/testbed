"""Unit tests for the GeodesicPolyhedron class in test.py (master branch).

These tests pin down the geometric invariants of the mesh generator:
vertex/face counts, unit-sphere placement, watertight manifold topology,
consistent outward winding, lazy-property caching/invalidation, and the
shared-midpoint behaviour that the subdivision algorithm depends on.

Known deviations from ideal behaviour are documented by strict xfail
tests in TestKnownIssuesOnMaster; each references the PR that fixes it.
"""

import gc
import weakref

import numpy as np
import pytest

from test import GeodesicPolyhedron


# Closed-form vertex/face counts for an icosahedron subdivided `d` times:
#   F(d) = 20 * 4**d ; V(d) = F(d)/2 + 2 (Euler: V - E + F = 2, E = 3F/2)
def expected_faces(divisions):
    return 20 * 4 ** divisions


def expected_vertices(divisions):
    return expected_faces(divisions) // 2 + 2


def edge_of(face_pair):
    """Canonical (sorted) edge tuple for one edge of a face."""
    a, b = face_pair
    return (a, b) if a < b else (b, a)


def build_edges(faces):
    """Map canonical edge -> list of face indices sharing it."""
    edges = {}
    for i, f in enumerate(faces):
        for e in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
            edges.setdefault(edge_of(e), []).append(i)
    return edges


@pytest.fixture(scope="module", params=[0, 1, 2, 3])
def poly(request):
    """One shared instance per subdivision level (construction is costly)."""
    return GeodesicPolyhedron(request.param)


@pytest.fixture()
def fresh():
    """A brand-new level-0 instance with untouched lazy caches.

    Needed because the module-scoped `poly` fixtures have their lazy
    properties computed by other tests within the same session.
    """
    return GeodesicPolyhedron(0)


# --------------------------------------------------------------------------
# Counts and basic structure
# --------------------------------------------------------------------------

class TestCounts:
    def test_face_count(self, poly):
        assert len(poly.getFaces()) == expected_faces(poly.divisions)

    def test_vertex_count(self, poly):
        assert len(poly.getVertices()) == expected_vertices(poly.divisions)

    def test_data_length_matches_faces(self, poly):
        assert len(poly.getData()) == len(poly.faces)

    @pytest.mark.parametrize("divisions", [0, 1, 2])
    def test_euler_characteristic_is_two(self, divisions):
        p = GeodesicPolyhedron(divisions)
        v = len(p.vertices)
        f = len(p.faces)
        e = len(build_edges(p.faces))
        assert v - e + f == 2

    def test_faces_are_index_triples_in_range(self, poly):
        n = len(poly.vertices)
        for face in poly.faces:
            assert isinstance(face, tuple) and len(face) == 3
            assert all(isinstance(i, int) for i in face)
            assert all(0 <= i < n for i in face)
            assert len(set(face)) == 3  # no degenerate faces


# --------------------------------------------------------------------------
# Geometry: everything on the unit sphere
# --------------------------------------------------------------------------

class TestGeometry:
    def test_all_vertices_on_unit_sphere(self, poly):
        radii = np.linalg.norm(np.asarray(poly.vertices), axis=1)
        assert np.allclose(radii, 1.0, atol=1e-12)

    def test_icosahedron_base_shape(self):
        p = GeodesicPolyhedron(0)
        assert len(p.vertices) == 12
        assert len(p.faces) == 20
        # The golden-ratio construction gives every vertex exactly one
        # zero coordinate and two non-zero coordinates.
        coords = np.abs(np.asarray(p.vertices))
        zeros = (coords < 1e-12).sum(axis=1)
        assert (zeros == 1).all()

    def test_midpoints_point_within_face_cone(self, poly):
        """Each stored midpoint must lie inside its own face's cone."""
        mps = poly.getFaceMidpoints()
        assert len(mps) == len(poly.faces)
        for mp, face in zip(mps, poly.faces):
            vs = np.asarray([poly.vertices[i] for i in face])
            for v in vs:
                assert float(np.dot(mp, v)) > 0


# --------------------------------------------------------------------------
# Topology: closed, manifold, consistently wound
# --------------------------------------------------------------------------

class TestTopology:
    def test_every_edge_shared_by_exactly_two_faces(self, poly):
        edges = build_edges(poly.faces)
        bad = [e for e, fs in edges.items() if len(fs) != 2]
        assert not bad, f"{len(bad)} non-manifold edges"

    def test_neighbour_lists_match_edge_sharing(self, poly):
        neighbours = poly.getNeighbours()
        assert len(neighbours) == len(poly.faces)
        edges = build_edges(poly.faces)
        expected = [[] for _ in poly.faces]
        for fs in edges.values():
            expected[fs[0]].append(fs[1])
            expected[fs[1]].append(fs[0])
        for i in range(len(poly.faces)):
            assert sorted(neighbours[i]) == sorted(expected[i])

    def test_each_face_has_three_distinct_neighbours(self, poly):
        for nb in poly.getNeighbours():
            assert len(nb) == 3
            assert len(set(nb)) == 3

    def test_neighbour_relation_is_symmetric(self, poly):
        neighbours = poly.getNeighbours()
        for i, nbrs in enumerate(neighbours):
            for j in nbrs:
                assert i in neighbours[j]

    def test_outward_consistent_winding(self, poly):
        """Every face normal must point away from the sphere centre."""
        vs = np.asarray(poly.vertices)
        for a, b, c in poly.faces:
            normal = np.cross(vs[b] - vs[a], vs[c] - vs[a])
            centroid = vs[a] + vs[b] + vs[c]
            assert float(np.dot(normal, centroid)) > 0

    def test_adjacent_faces_use_opposite_edge_directions(self, poly):
        """Watertight orientation: shared edge traversed opposite ways."""
        directed = {}
        for face in poly.faces:
            for x, y in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
                directed[(x, y)] = directed.get((x, y), 0) + 1
        for (x, y), count in directed.items():
            assert count == 1
            assert (y, x) in directed  # reverse arc also exists exactly once


# --------------------------------------------------------------------------
# Subdivision mechanics: shared midpoints & data inheritance
# --------------------------------------------------------------------------

class TestSubdivisionMechanics:
    def test_shared_edges_reuse_one_midpoint_vertex(self):
        """One midpoint per base edge (30), no duplicates, watertight."""
        p = GeodesicPolyhedron(1)
        new = np.asarray(p.vertices[12:])
        assert len(new) == 30  # exactly one appended vertex per base edge
        dists = np.linalg.norm(new[:, None, :] - new[None, :, :], axis=-1)
        iu = np.triu_indices(len(new), k=1)
        assert dists[iu].min() > 1e-6  # all distinct
        # And the resulting mesh is still closed/manifold:
        assert all(len(fs) == 2 for fs in build_edges(p.faces).values())

    def test_midpoint_lookup_is_order_independent(self, fresh):
        before = len(fresh.vertices)
        i = fresh._createMiddlePointVertex(5, 9)
        j = fresh._createMiddlePointVertex(9, 5)
        assert i == j
        assert len(fresh.vertices) == before + 1  # only one vertex added

    def test_midpoint_vertex_lies_on_great_circle_arc(self, fresh):
        a, b = 0, 11
        idx = fresh._createMiddlePointVertex(a, b)
        m = fresh.vertices[idx]
        va, vb = fresh.vertices[a], fresh.vertices[b]
        expected = (va + vb) / 2
        expected = expected / np.linalg.norm(expected)
        assert np.allclose(m, expected)

    def test_face_data_propagates_to_children(self, fresh):
        fresh.data[0] = "north-cap"
        fresh._subdivideFaces()
        children = [d for d in fresh.data if d == "north-cap"]
        assert len(children) == 4
        # Same object referenced, not copies.
        assert len({id(d) for d in children}) == 1

    def test_subdivision_quadruples_faces(self, fresh):
        f0 = len(fresh.faces)
        fresh._subdivideFaces()
        assert len(fresh.faces) == 4 * f0


# --------------------------------------------------------------------------
# Lazy properties: caching and invalidation
# --------------------------------------------------------------------------

class TestLazyProperties:
    def test_midpoints_lazy_then_cached(self, fresh):
        assert fresh.facemidpoints is None
        first = fresh.getFaceMidpoints()
        assert fresh.facemidpoints is first
        assert fresh.getFaceMidpoints() is first  # same list object

    def test_neighbours_lazy_then_cached(self, fresh):
        assert fresh.neighbours is None
        first = fresh.getNeighbours()
        assert fresh.neighbours is first
        assert fresh.getNeighbours() is first

    def test_getters_return_live_lists(self, poly):
        assert poly.getVertices() is poly.vertices
        assert poly.getFaces() is poly.faces
        assert poly.getData() is poly.data

    def test_subdivision_invalidates_caches(self):
        p = GeodesicPolyhedron(1)
        p.getFaceMidpoints()
        p.getNeighbours()
        p._subdivideFaces()
        assert p.facemidpoints is None
        assert p.neighbours is None
        # And they regenerate correctly afterwards.
        assert len(p.getFaceMidpoints()) == len(p.faces)
        assert len(p.getNeighbours()) == len(p.faces)


# --------------------------------------------------------------------------
# Instance isolation
# --------------------------------------------------------------------------

class TestInstanceIsolation:
    def test_instances_have_independent_meshes(self):
        a = GeodesicPolyhedron(1)
        b = GeodesicPolyhedron(2)
        assert len(a.faces) == 80 and len(b.faces) == 320
        assert a.vertices is not b.vertices
        a.data[0] = "only-in-a"
        assert b.data[0] is None

    def test_mutating_one_instance_does_not_affect_other(self):
        a = GeodesicPolyhedron(1)
        b = GeodesicPolyhedron(1)
        n_before = len(b.vertices)
        a._createMiddlePointVertex(0, 1)
        assert len(b.vertices) == n_before


# --------------------------------------------------------------------------
# Robustness / input handling
# --------------------------------------------------------------------------

class TestRobustness:
    def test_negative_divisions_yields_plain_icosahedron(self):
        p = GeodesicPolyhedron(-1)
        assert len(p.faces) == 20
        assert len(p.vertices) == 12

    @pytest.mark.xfail(reason="KNOWN ISSUE: floats are accepted silently "
                              "(range() treats them like ints); validation "
                              "should reject them.")
    def test_float_divisions_should_be_rejected(self):
        with pytest.raises(TypeError):
            GeodesicPolyhedron(1.5)


# --------------------------------------------------------------------------
# Known issues on master (documented failures, strict so they flip to
# XPASS->failure when the corresponding fix PRs are merged)
# --------------------------------------------------------------------------

class TestKnownIssuesOnMaster:
    @pytest.mark.xfail(strict=True, reason="KNOWN ISSUE: functools.cache is "
                       "class-level; it pins instances so they are never "
                       "garbage-collected (memory leak). Fixed in PR #1.")
    def test_instances_are_garbage_collectable(self):
        p = GeodesicPolyhedron(1)
        ref = weakref.ref(p)
        del p
        gc.collect()
        assert ref() is None

    def test_class_level_cache_grows_across_instances(self):
        """Evidence for the class-level cache design flaw (fixed in PR #1):
        functools.cache is bound at CLASS level, so every instance shares
        one unbounded cache whose entries accumulate forever. Because the
        cache key includes `self`, cross-instance contamination of RESULTS
        does not occur; the real damage is instance pinning (GC leak, tested
        above) and unbounded growth that no instance can clear.
        """
        if not hasattr(GeodesicPolyhedron._createMiddlePointVertex, "cache_info"):
            pytest.skip("per-instance cache in use (PR #1 merged) - issue fixed")
        before = GeodesicPolyhedron._createMiddlePointVertex.cache_info().currsize
        p = GeodesicPolyhedron(1)  # builds ~150 midpoints through the shared cache
        after = GeodesicPolyhedron._createMiddlePointVertex.cache_info().currsize
        assert after > before      # class-level cache grew with this instance's keys
        del p
        gc.collect()
        # Entries survive the instance's death (they pin it via `self` key)...
        assert (GeodesicPolyhedron._createMiddlePointVertex
                .cache_info().currsize == after)
        if not hasattr(GeodesicPolyhedron._createMiddlePointVertex, "cache_clear"):
            pytest.skip("no class-level cache to clear (PR #1 merged)")
        GeodesicPolyhedron._createMiddlePointVertex.cache_clear()  # cleanup

    @pytest.mark.xfail(strict=True, reason="KNOWN ISSUE: face midpoints are "
                       "raw centroids, NOT re-projected onto the sphere "
                       "(radii < 1). Fixed by the midpoint re-projection "
                       "commits (PRs #1/#2/#3).")
    def test_face_midpoints_lie_on_unit_sphere(self):
        p = GeodesicPolyhedron(3)
        radii = np.linalg.norm(np.asarray(p.getFaceMidpoints()), axis=1)
        assert np.allclose(radii, 1.0, atol=1e-12)

    def test_unprojected_centroids_measurably_inside_sphere(self):
        """Companion check proving the xfail above is genuine: coarse
        faces have centroids measurably inside the sphere (~0.795 for
        icosahedron faces)."""
        p = GeodesicPolyhedron(0)
        radii = np.linalg.norm(np.asarray(p.getFaceMidpoints()), axis=1)
        assert radii.max() < 1.0 - 1e-3
