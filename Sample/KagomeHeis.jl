include("../src/MeanFieldToolkit.jl")
using .MeanFieldToolkit

using LinearAlgebra, TightBindingToolkit, Distributions, LaTeXStrings
##### Kagome lattice
const a1  =   [4.0, 0.0]
const a2  =   [1.0, sqrt(3)]

const b1  =   [0.0, 0.0]
const b2  =   [1.0, 0.0] 
const b3  =   [0.5, sqrt(3)/2]  
const b4  =   [2.0 + 0.0, 0.0]
const b5  =   [2.0 + 1.0, 0.0] 
const b6  =   [2.0 + 0.5, sqrt(3)/2]

const InitialField  =   zeros(Float64, 4)
const OnSite        =   SpinMats(1//2)

HoppingUC       =   UnitCell([a1, a2], 2, 2)
InteractionUC   =   UnitCell([a1, a2], 2, 4)
AddBasisSite!.(Ref(HoppingUC),     [b1, b2, b3, b4, b5, b6], Ref(InitialField) , Ref(OnSite))
AddBasisSite!.(Ref(InteractionUC), [b1, b2, b3, b4, b5, b6], Ref(InitialField) , Ref(OnSite))

const J     =   1.0
Jmatrix     =   Matrix{Float64}(I, 3, 3)
U           =   SpinToPartonCoupling(Jmatrix, 1//2)
###### Interaction Block                                  
JParam      =   Param(J, 4)
AddIsotropicBonds!(JParam, InteractionUC, 1.0, U, "Heisenberg Interaction")
InteractionBlock   =  ParamBlock([JParam], InteractionUC)

const n       =   20
const kSize   =   6 * n + 3  
bz            =   BZ([kSize, kSize])
FillBZ!(bz, HoppingUC)

##### Thermodynamic parameters
const T         =   0.001
const filling   =   0.5
const stat      =   -1

const mixingAlpha    =    0.5  

phis            =   collect(LinRange(-pi/2, pi/2, 21))

for ϕ in phis

    ChiUC           =   UnitCell([a1, a2], 2, 2)

    AddBasisSite!.(Ref(ChiUC), [b1, b2, b3, b4, b5, b6], Ref(InitialField), Ref(OnSite))

    ##### Hopping expectation params
    # Hopping with flux ϕ through the triangles, and π-2ϕ through the hexagons
    t_flux  =   Param(1.0, 2)
    AddAnisotropicBond!(t_flux, ChiUC, 1, 2, [  0,  0],  exp( im * ϕ/3) * OnSite[4], 1.0, "[ϕ, ϕ, π - 2*ϕ] Hopping with ϕ/π = $(ϕ/pi)")
    AddAnisotropicBond!(t_flux, ChiUC, 2, 3, [  0,  0],  exp( im * ϕ/3) * OnSite[4], 1.0, "[ϕ, ϕ, π - 2*ϕ] Hopping with ϕ/π = $(ϕ/pi)")
    AddAnisotropicBond!(t_flux, ChiUC, 3, 1, [  0,  0],  exp( im * ϕ/3) * OnSite[4], 1.0, "[ϕ, ϕ, π - 2*ϕ] Hopping with ϕ/π = $(ϕ/pi)")
    AddAnisotropicBond!(t_flux, ChiUC, 4, 5, [  0,  0],  exp( im * ϕ/3) * OnSite[4], 1.0, "[ϕ, ϕ, π - 2*ϕ] Hopping with ϕ/π = $(ϕ/pi)")
    AddAnisotropicBond!(t_flux, ChiUC, 5, 6, [  0,  0],  exp( im * ϕ/3) * OnSite[4], 1.0, "[ϕ, ϕ, π - 2*ϕ] Hopping with ϕ/π = $(ϕ/pi)")
    AddAnisotropicBond!(t_flux, ChiUC, 6, 4, [  0,  0],  exp( im * ϕ/3) * OnSite[4], 1.0, "[ϕ, ϕ, π - 2*ϕ] Hopping with ϕ/π = $(ϕ/pi)")
    AddAnisotropicBond!(t_flux, ChiUC, 4, 2, [  0,  0],  exp( im * ϕ/3) * OnSite[4], 1.0, "[ϕ, ϕ, π - 2*ϕ] Hopping with ϕ/π = $(ϕ/pi)")

    AddAnisotropicBond!(t_flux, ChiUC, 2, 6, [  0, -1],  exp( im * (pi + ϕ/3)) * OnSite[4], 1.0, "[ϕ, ϕ, π - 2*ϕ] Hopping with ϕ/π = $(ϕ/pi)")
    AddAnisotropicBond!(t_flux, ChiUC, 4, 6, [  0, -1],  exp(-im * (pi + ϕ/3)) * OnSite[4], 1.0, "[ϕ, ϕ, π - 2*ϕ] Hopping with ϕ/π = $(ϕ/pi)")
    AddAnisotropicBond!(t_flux, ChiUC, 5, 3, [  1, -1],  exp( im * (pi + ϕ/3)) * OnSite[4], 1.0, "[ϕ, ϕ, π - 2*ϕ] Hopping with ϕ/π = $(ϕ/pi)")
    AddAnisotropicBond!(t_flux, ChiUC, 5, 1, [  1,  0],  exp(-im * (pi + ϕ/3)) * OnSite[4], 1.0, "[ϕ, ϕ, π - 2*ϕ] Hopping with ϕ/π = $(ϕ/pi)")
    AddAnisotropicBond!(t_flux, ChiUC, 3, 1, [  0,  1],  exp( im * ϕ/3) * OnSite[4], 1.0, "[ϕ, ϕ, π - 2*ϕ] Hopping with ϕ/π = $(ϕ/pi)")

    ######### Weiss fields
    Ordering    =   Param(1.0, 2)
    AddAnisotropicBond!(Ordering, ChiUC, 1, 1, [ 0, 0],  sum([ -sqrt(3)/2, -1/2, 0.0] .* OnSite[1:3]), 0.0, "120 degree ordering")
    AddAnisotropicBond!(Ordering, ChiUC, 2, 2, [ 0, 0],  sum([  sqrt(3)/2, -1/2, 0.0] .* OnSite[1:3]), 0.0, "120 degree ordering")
    AddAnisotropicBond!(Ordering, ChiUC, 3, 3, [ 0, 0],  sum([        0.0,  1.0, 0.0] .* OnSite[1:3]), 0.0, "120 degree ordering")
    AddAnisotropicBond!(Ordering, ChiUC, 4, 4, [ 0, 0],  sum([ -sqrt(3)/2, -1/2, 0.0] .* OnSite[1:3]), 0.0, "120 degree ordering")
    AddAnisotropicBond!(Ordering, ChiUC, 5, 5, [ 0, 0],  sum([  sqrt(3)/2, -1/2, 0.0] .* OnSite[1:3]), 0.0, "120 degree ordering")
    AddAnisotropicBond!(Ordering, ChiUC, 6, 6, [ 0, 0],  sum([        0.0,  1.0, 0.0] .* OnSite[1:3]), 0.0, "120 degree ordering")

    ChiParams    =  [t_flux, Ordering]
    HoppingBlock =  ParamBlock(ChiParams, ChiUC)

    H             =   Hamiltonian(HoppingUC, bz)
    DiagonalizeHamiltonian!(H)
    M             =   Model(HoppingUC, bz, H ; T=T, filling=filling, stat=stat)

    mft            =   TightBindingMFT(M, HoppingBlock, InteractionBlock, InterQuarticToHopping, Dict{String, Float64}("ij" => 1.0, "ii" => 0.0, "jj" => 0.0) )
    fileName    =   "./Sample/KagomeDiracSL/J=$(round(J, digits=3))_phi=$(round(ϕ/pi, digits=3))Pi_New.jld2"
    SolveMFT!(mft, fileName)

end
