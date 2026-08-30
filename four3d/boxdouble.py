from netgen.occ import *


def create_geometry_3d(delta, maxh):
    """
    Versione 3D della mesh 2D:
      - dominio: [0, delta] x [0, 1] x [0, 1]
      - 4 inlet circolari sulla faccia x=0
      - 4 outlet circolari sulla faccia x=delta
      - restanti facce del box marcate come pareti

    I 4 inlet/outlet sono disposti in una griglia 2x2 sulle facce x=0 e x=delta.
    """

    Lx = delta
    Ly = 1.0
    Lz = 1.0

    # Diametro degli ingressi/uscite, ispirato alla scala della mesh 2D
    d = 1.0 / 6.0
    r = d / 2.0

    # posizioni dei centri sulla faccia y-z
    pos = [0.3, 0.7]

    box = Box((0, 0, 0), (Lx, Ly, Lz))

    # nomi delle facce del box
    box.faces.Min(X).name = "wall_xmin"
    box.faces.Max(X).name = "wall_xmax"
    box.faces.Min(Y).name = "wall_ymin"
    box.faces.Max(Y).name = "wall_ymax"
    box.faces.Min(Z).name = "wall_zmin"
    box.faces.Max(Z).name = "wall_zmax"

    # ------------------------------------------------------------
    # 4 inlet su x = 0
    # ------------------------------------------------------------
    inlets = []
    inlet_id = 1
    for y0 in pos:
        for z0 in pos:
            center = Pnt(0.0, y0, z0)
            inlet = WorkPlane(
                Axes(center, n=Dir(-1, 0, 0), h=Dir(0, 1, 0))
            ).Circle(r).Face()
            inlet.faces.name = f"inlet{inlet_id}"
            inlets.append(inlet)
            inlet_id += 1

    # ------------------------------------------------------------
    # 4 outlet su x = Lx
    # ------------------------------------------------------------
    outlets = []
    outlet_id = 1
    for y0 in pos:
        for z0 in pos:
            center = Pnt(Lx, y0, z0)
            outlet = WorkPlane(
                Axes(center, n=Dir(1, 0, 0), h=Dir(0, 1, 0))
            ).Circle(r).Face()
            outlet.faces.name = f"outlet{outlet_id}"
            outlets.append(outlet)
            outlet_id += 1

    # unione geometrica
    shape = Glue([box] + inlets + outlets)

    geo = OCCGeometry(shape, dim=3)
    ngmesh = geo.GenerateMesh(maxh=maxh)

    names = ngmesh.GetRegionNames(codim=1)
    markers = {name: i + 1 for i, name in enumerate(names)}

    return ngmesh, markers