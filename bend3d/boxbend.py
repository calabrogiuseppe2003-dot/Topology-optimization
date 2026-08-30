from netgen.occ import *


def mesh_bend_pipe_3d(maxh, l=1.0):
    """Cubo [0,l]^3 con:
       - inlet circolare sulla faccia x=0, centro (0, l/2, 4l/5), diametro l/5
       - outlet circolare sulla faccia z=0, centro (4l/5, l/2, 0), diametro l/5
    """

    r = (l / 5) / 2.0  # raggio = diametro/2

    box = Box((0, 0, 0), (l, l, l))

    # Nomi di default per tutte le facce del cubo (verranno parzialmente
    # sovrascritti dal Glue sulle facce x=0 e z=0).
    box.faces.Min(X).name = "wall_xmin"
    box.faces.Max(X).name = "wall_xmax"
    box.faces.Min(Y).name = "wall_ymin"
    box.faces.Max(Y).name = "wall_ymax"
    box.faces.Min(Z).name = "wall_zmin"
    box.faces.Max(Z).name = "wall_zmax"

    # ---- Inlet: disco sulla faccia x = 0 ----
    inlet_center = Pnt(0.0, l / 2, 4 * l / 5)
    inlet = WorkPlane(
        Axes(inlet_center, n=Dir(-1, 0, 0), h=Dir(0, 1, 0))
    ).Circle(r).Face()
    inlet.faces.name = "inlet"

    # ---- Outlet: disco sulla faccia z = 0 ----
    outlet_center = Pnt(4 * l / 5, l / 2, 0.0)
    outlet = WorkPlane(
        Axes(outlet_center, n=Dir(0, 0, -1), h=Dir(1, 0, 0))
    ).Circle(r).Face()
    outlet.faces.name = "outlet"

    shape = Glue([box, inlet, outlet])

    geo = OCCGeometry(shape, dim=3)
    ngmesh = geo.GenerateMesh(maxh=maxh)
    names = ngmesh.GetRegionNames(codim=1)

    markers = {
        name: i + 1
        for i, name in enumerate(names)
    }

    return ngmesh, markers


#if __name__ == "__main__":
    # ngmesh, markers = mesh_bend_pipe_3d(0.05)
    # mesh = Mesh(ngmesh)
    # vtk = VTKOutput(
    #     ma=mesh,
    #     coefs=[],
    #     names=[],
    #     filename="mesh",
    #     subdivision=0
    # )

    # vtk.Do()