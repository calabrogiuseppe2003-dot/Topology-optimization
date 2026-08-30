from netgen.geom2d import SplineGeometry

def create_geometry(delta,maxh):
    Lx = delta
    Ly = 1.0
    l  = 1.0/6.0

    y1a = 0.25 - l/2
    y1b = 0.25 + l/2

    y2a = 0.75 - l/2
    y2b = 0.75 + l/2

    geo = SplineGeometry()

    # ------------------------------------------------------------------
    # punti lato sinistro
    # ------------------------------------------------------------------

    p1 = geo.AppendPoint(0.0, 0.0)
    p2 = geo.AppendPoint(0.0, y1a)
    p3 = geo.AppendPoint(0.0, y1b)
    p4 = geo.AppendPoint(0.0, y2a)
    p5 = geo.AppendPoint(0.0, y2b)
    p6 = geo.AppendPoint(0.0, Ly)

    # ------------------------------------------------------------------
    # punti lato destro
    # ------------------------------------------------------------------

    p7  = geo.AppendPoint(Lx, 0.0)
    p8  = geo.AppendPoint(Lx, y1a)
    p9  = geo.AppendPoint(Lx, y1b)
    p10 = geo.AppendPoint(Lx, y2a)
    p11 = geo.AppendPoint(Lx, y2b)
    p12 = geo.AppendPoint(Lx, Ly)

    # ------------------------------------------------------------------
    # bordo antiorario
    # ------------------------------------------------------------------

    # lato inferiore
    geo.Append(
        ["line", p1, p7],
        leftdomain=1,
        rightdomain=0,
        bc="bottom"
    )

    # lato destro

    geo.Append(
        ["line", p7, p8],
        leftdomain=1,
        rightdomain=0,
        bc="right_wall_bottom"
    )

    geo.Append(
        ["line", p8, p9],
        leftdomain=1,
        rightdomain=0,
        bc="right_outlet1"
    )

    geo.Append(
        ["line", p9, p10],
        leftdomain=1,
        rightdomain=0,
        bc="right_wall_middle"
    )

    geo.Append(
        ["line", p10, p11],
        leftdomain=1,
        rightdomain=0,
        bc="right_outlet2"
    )

    geo.Append(
        ["line", p11, p12],
        leftdomain=1,
        rightdomain=0,
        bc="right_wall_top"
    )

    # lato superiore

    geo.Append(
        ["line", p12, p6],
        leftdomain=1,
        rightdomain=0,
        bc="top"
    )

    # lato sinistro

    geo.Append(
        ["line", p6, p5],
        leftdomain=1,
        rightdomain=0,
        bc="left_wall_top"
    )

    geo.Append(
        ["line", p5, p4],
        leftdomain=1,
        rightdomain=0,
        bc="left_inlet2"
    )

    geo.Append(
        ["line", p4, p3],
        leftdomain=1,
        rightdomain=0,
        bc="left_wall_middle"
    )

    geo.Append(
        ["line", p3, p2],
        leftdomain=1,
        rightdomain=0,
        bc="left_inlet1"
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
