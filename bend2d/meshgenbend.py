from netgen.geom2d import SplineGeometry

def create_geometry_bend(maxh):
    Lx = 1.0
    Ly = 1.0
    l  = 1.0/5.0

    y1a = 0.8 - l/2
    y1b = 0.8 + l/2

    x2a = 0.8 - l/2
    x2b = 0.8 + l/2

    geo = SplineGeometry()

    # ------------------------------------------------------------------
    # punti lato sinistro
    # ------------------------------------------------------------------

    p1 = geo.AppendPoint(0.0, 0.0)
    p2 = geo.AppendPoint(0.0, y1a)
    p3 = geo.AppendPoint(0.0, y1b)
    p4 = geo.AppendPoint(0.0, Ly)

    # ------------------------------------------------------------------
    # punti lato destro
    # ------------------------------------------------------------------

    p5  = geo.AppendPoint(Lx, Ly)
    p6 = geo.AppendPoint(Lx, 0.0)

    # ------------------------------------------------------------------
    # punti lato inferiore
    # ------------------------------------------------------------------

    p7 = geo.AppendPoint(x2b, 0.0)
    p8 = geo.AppendPoint(x2a, 0.0)

    # ------------------------------------------------------------------
    # bordo antiorario
    # ------------------------------------------------------------------

    # lato inferiore
    geo.Append(
        ["line", p1, p8],
        leftdomain=1,
        rightdomain=0,
        bc="bottom_wall_left"
    )

    geo.Append(
        ["line", p8, p7],
        leftdomain=1,
        rightdomain=0,
        bc="bottom_outlet"
    )

    geo.Append(
        ["line", p7, p6],
        leftdomain=1,
        rightdomain=0,
        bc="bottom_wall_right"
    )

    # lato destro

    geo.Append(
        ["line", p6, p5],
        leftdomain=1,
        rightdomain=0,
        bc="right_wall"
    )

    # lato superiore

    geo.Append(
        ["line", p5, p4],
        leftdomain=1,
        rightdomain=0,
        bc="top_wall"
    )

    # lato sinistro

    geo.Append(
        ["line", p4, p3],
        leftdomain=1,
        rightdomain=0,
        bc="left_wall_top"
    )

    geo.Append(
        ["line", p3, p2],
        leftdomain=1,
        rightdomain=0,
        bc="left_inlet"
    )

    geo.Append(
        ["line", p2, p1],
        leftdomain=1,
        rightdomain=0,
        bc="left_wall_bottom"
    )

    # ------------------------------------------------------------------
    # mesh
    # ------------------------------------------------------------------

    ngmesh = geo.GenerateMesh(maxh=maxh)
    names = ngmesh.GetRegionNames(codim=1)

    markers = {
        name: i+1
        for i, name in enumerate(names)
    }

    return ngmesh, markers