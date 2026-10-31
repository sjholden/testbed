import numpy as np
import functools
import scipy as sp
import scipy.spatial

class GeodesicPolyhedron:
    """
    A class generating a geodesic polyhedron of a unit sphere.

    Attributes:
        divisions (int): The number of triangle subdivisions performed.
        vertices (list(point): The triangle vertices.
        faces (list(tuple(int,int,int)): The triangles as three indices to the
                                         vertices list.
        facemidpoints (list(point)): The midpoint of the faces - indices align 
                                     with the faces list.
        neighbours (list(list(int)): The neighbouring traingles. Indices align 
                                     with the faces list, values are indices
                                     into the faces list.
        data (list(?)): User determined data associated with the faces, if 
                        a face is subdivided the data will be referenced by 
                        all the resulting faces.
    """

    def __init__(self, divisions: int):
        """
        Constructor.

        Arguments:
            divisions(int): Number of times to subdivide triangles.
        """
        self.divisions = divisions
        self._generateIcosohedron()
        self._generateGeodesic()

    def getVertices(self):
        """Return the list of vertices."""
        return self.vertices

    def getFaces(self):
        """Return the list of faces."""
        return self.faces

    def getData(self):
        """Return the list of face data."""
        return self.data

    def getFaceMidpoints(self):
        """
        Return the list of midpoints.

        Will generate the midpoints if they have not already been calculated.
        """
        if self.facemidpoints is None:
            self._generateFaceMidpoints()
        return self.facemidpoints

    def getNeighbours(self):
        """
        Return the list of neighbours.

        Will generate the neighbours if they have not already been calculated.
        """
        if self.neighbours is None:
            self._generateNeighbours()
        return self.neighbours

    def _generateIcosohedron(self):
        """Generate the initial icosohedron."""
        gr = (1 + np.sqrt(5)) / 2
        norm = np.linalg.norm((gr,1))
        one = 1/norm
        gr = gr/norm
        self.vertices = list(map(np.array,[(-one,gr,0), (one,gr,0), (-one,-gr,0),
                                           (one,-gr,0), (0,-one,gr), (0,one,gr),
                                           (0,-one,-gr), (0,one,-gr), (gr,0,-one),
                                           (gr,0,one), (-gr,0,-one), (-gr,0,one)
                                          ]))
        self.faces = [
                (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
                (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
                (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
                (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)
            ]
        self.data = [None for x in self.faces]
        self.facemidpoints = None
        self.neighbours = None

    def _generateGeodesic(self):
        """Convert the icosohedron into a geodesic polyhedron."""
        for _ in range(self.divisions):
            self._subdivideFaces()

    def _subdivideFaces(self):
        """Subdivide every triangle into four triangles."""
        newfaces = []
        newdata = []
        data = self.data
        for index in range(len(self.faces)):
            face = self.faces[index]
            data = self.data[index]
            a = self._createMiddlePointVertex(face[0], face[1])
            b = self._createMiddlePointVertex(face[1], face[2])
            c = self._createMiddlePointVertex(face[2], face[0])
            newfaces.append((face[0], a, c))
            newfaces.append((face[1], b, a))
            newfaces.append((face[2], c, b))
            newfaces.append((a, b, c))
            for _ in range(4):
                newdata.append(data)
        self.faces = newfaces
        self.data = newdata
        self.facemidpoints = None
        self.neighbours = None

    @functools.cache
    def _createMiddlePointVertex(self, a, b):
        """
        Create a new vertex at the midpoint of two existing vertices.

        @cache is not for performance, it is essential to the algorithm.
        The midpoint of a and b is determined and normalized to be on
        the sphere. If that midpoint was already generated the existing
        index is used.

        Arguments:
            a(int): Index of the first vertex.
            b(int): Index of the second vertex.

        Returns:
            (int): The index of the middle point vertex.
        """

        if b < a:
            return self._createMiddlePointVertex(b, a)
        vertex = (self.vertices[a] + self.vertices[b]) / 2
        vertex = vertex / np.linalg.norm(vertex)
        self.vertices.append(vertex)
        return len(self.vertices)-1

    def _generateFaceMidpoints(self):
        """
        Calculate and store the midpoints of the triangle faces.

        Note: Not actually the midpoint - this assumes the triangles are
        equilateral, but due to the projection on the sphere they are not.
        """
        midpoints = []
        vertices = self.vertices
        for face in self.faces:
            midpoints.append((vertices[face[0]] + vertices[face[1]] + vertices[face[2]])/3)
        self.facemidpoints = midpoints

    def _generateNeighbours(self):
        """Calculate and store a list of neighbours for each face."""
        edges = {}
        def addFaceToEdges(a, b, face):
            if b<a:
                b,a = a,b
            edges.setdefault((a,b),[]).append(face)
        for face in range(len(self.faces)):
            addFaceToEdges(self.faces[face][0], self.faces[face][1], face)
            addFaceToEdges(self.faces[face][0], self.faces[face][2], face)
            addFaceToEdges(self.faces[face][1], self.faces[face][2], face)
        neighbours = [[] for x in self.faces]
        for faces in edges.values():
            neighbours[faces[0]].append(faces[1])
            neighbours[faces[1]].append(faces[0])
        self.neighbours = neighbours
            


        
    

if __name__ == '__main__':
    i = GeodesicPolyhedron(1)
    print(i.vertices)
    print(i.faces)
    #import matplotlib.pyplot as plt
    #fig = plt.figure()
    #ax = fig.add_subplot(projection='3d')
    #for v in i.vertices:
    #    ax.scatter(v[0], v[1], v[2])
    #plt.show()
    mp = i.getFaceMidpoints()
    tree = sp.spatial.KDTree(mp)
    import random
    for _ in range(100):
        v = np.array([random.random(),random.random(),random.random()])
        v = v/ np.linalg.norm(v)
        dd,ii = tree.query([v])
        print(tree.data[ii[0]])
        print(mp[ii[0]])
        #print(v +  "->" + repr([dd,ii]))
    print(i.getNeighbours())
        
    

