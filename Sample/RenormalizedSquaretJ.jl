include("../src/MeanFieldToolkit.jl")
using .MeanFieldToolkit

using LinearAlgebra, TightBindingToolkit, Distributions, Base.Threads

##### Square lattice
const a1  =   [1.0, 1.0]
const a2  =   [1.0, -1.0]

const b1  =   [0.0, 0.0]
const b2  =   [1.0, 0.0]

const InitialField    =   zeros(Float64, 4)

const n       =   10
const kSize   =   6 * n + 3 

SpinVec     =   SpinMats(1//2)
const t     =   1.0

##### Thermodynamic parameters
const T         =   0.001
const stat      =   -1

const mixingAlpha    =    0.5   

##### Renormalized MFT from https://arxiv.org/pdf/cond-mat/0311604.pdf

fillings = collect(LinRange(0.2, 0.49, 16))

for filling in fillings
# filling     =   0.45
    delta = 1-2*filling
    rnorm_hopping_factor = 2*delta/(1+delta)
    rnorm_int_factor = 4/((1+delta)^2)

    HoppingUC       =   UnitCell([a1, a2], 2, 2)
    PairingUC       =   UnitCell([a1, a2], 2, 2)
    ChiUC           =   UnitCell([a1, a2], 2, 2)
    DeltaUC         =   UnitCell([a1, a2], 2, 2)
    InteractionUC   =   UnitCell([a1, a2], 2, 4)

    AddBasisSite!(HoppingUC, b1, InitialField)
    AddBasisSite!(HoppingUC, b2, InitialField)

    AddBasisSite!(PairingUC, b1, InitialField)
    AddBasisSite!(PairingUC, b2, InitialField)

    AddBasisSite!(ChiUC, b1, InitialField)
    AddBasisSite!(ChiUC, b2, InitialField)

    AddBasisSite!(DeltaUC, b1, InitialField)
    AddBasisSite!(DeltaUC, b2, InitialField)

    AddBasisSite!(InteractionUC, b1, InitialField)
    AddBasisSite!(InteractionUC, b2, InitialField)

    ##### HoppingParams
    t1          =   -t*rnorm_hopping_factor
    t1Param     =   Param(t1, 2)
    AddIsotropicBonds!(t1Param, HoppingUC, 1.0, SpinVec[4], "t1")

    HoppingParams   =   [t1Param]
    CreateUnitCell!(HoppingUC, HoppingParams)

    ##### Hopping expectation params
    t_s    =   Param(1.0, 2)
    AddIsotropicBonds!(t_s, ChiUC, 1.0, SpinVec[4], "s Hopping")

    t_px    =  Param(1.0, 2)
    AddAnisotropicBond!(t_px, ChiUC, 1, 2, [ 0,  0],  im * SpinVec[4], 1.0, "p_x Hopping")
    AddAnisotropicBond!(t_px, ChiUC, 1, 2, [-1, -1], -im * SpinVec[4], 1.0, "p_x Hopping")

    t_py    =  Param(1.0, 2)
    AddAnisotropicBond!(t_py, ChiUC, 1, 2, [ 0,  -1],  im * SpinVec[4], 1.0, "p_y Hopping")
    AddAnisotropicBond!(t_py, ChiUC, 1, 2, [ -1,  0], -im * SpinVec[4], 1.0, "p_y Hopping")

    t_dx2y2 =   Param(1.0, 2)
    AddAnisotropicBond!(t_dx2y2, ChiUC, 1, 2, [ 0,   0],  SpinVec[4], 1.0, "d_x^2-y^2 Hopping")
    AddAnisotropicBond!(t_dx2y2, ChiUC, 1, 2, [-1 , -1],  SpinVec[4], 1.0, "d_x^2-y^2 Hopping")
    AddAnisotropicBond!(t_dx2y2, ChiUC, 1, 2, [ 0,  -1], -SpinVec[4], 1.0, "d_x^2-y^2 Hopping")
    AddAnisotropicBond!(t_dx2y2, ChiUC, 1, 2, [ -1,  0], -SpinVec[4], 1.0, "d_x^2-y^2 Hopping")

    Neel_Weiss    =   Param(1.0, 2)
    AddAnisotropicBond!(Neel_Weiss, ChiUC, 1, 1, [ 0, 0],   SpinVec[3], 0.0, "Neel Weiss")
    AddAnisotropicBond!(Neel_Weiss, ChiUC, 2, 2, [ 0, 0],  -SpinVec[3], 0.0, "Neel Weiss")

    ChiParams    =  [t_s]
    HoppingBlock =  ParamBlock(ChiParams, ChiUC)

    ##### Pairing expectation params
    p_s    =   Param(1.0, 2)
    AddIsotropicBonds!(p_s, DeltaUC, 1.0, SpinVec[2], "s Pairing")

    p_px    =   Param(1.0, 2)
    AddAnisotropicBond!(p_px, DeltaUC, 1, 2, [ 0,  0],  im * (SpinVec[1]), 1.0, "p_x Pairing")
    AddAnisotropicBond!(p_px, DeltaUC, 1, 2, [-1, -1], -im * (SpinVec[1]), 1.0, "p_x Pairing")

    p_py    =   Param(1.0, 2)
    AddAnisotropicBond!(p_py, DeltaUC, 1, 2, [ 0,  -1],   im * (SpinVec[1]), 1.0, "p_y Pairing")
    AddAnisotropicBond!(p_py, DeltaUC, 1, 2, [ -1,  0],  -im * (SpinVec[1]), 1.0, "p_y Pairing")

    p_dx2y2 =   Param(1.0, 2)
    AddAnisotropicBond!(p_dx2y2, DeltaUC, 1, 2, [ 0,   0],  (SpinVec[2]), 1.0, "d_x^2-y^2 Pairing")
    AddAnisotropicBond!(p_dx2y2, DeltaUC, 1, 2, [-1 , -1],  (SpinVec[2]), 1.0, "d_x^2-y^2 Pairing")
    AddAnisotropicBond!(p_dx2y2, DeltaUC, 1, 2, [ 0,  -1], -(SpinVec[2]), 1.0, "d_x^2-y^2 Pairing")
    AddAnisotropicBond!(p_dx2y2, DeltaUC, 1, 2, [ -1,  0], -(SpinVec[2]), 1.0, "d_x^2-y^2 Pairing")

    DeltaParams    =  [p_dx2y2]
    PairingBlock   =  ParamBlock(DeltaParams, DeltaUC)

    ###### Interaction Block                                  
    J           =   (t/5) * rnorm_int_factor
    Jmatrix     =   Matrix{Float64}(I, 3, 3)
    U           =   SpinToPartonCoupling(Jmatrix, 1//2)
    JParam      =   Param(J, 4)
    AddIsotropicBonds!(JParam, InteractionUC, 1.0, U, "Heisenberg Interaction")
    InteractionBlock   =  ParamBlock([JParam], InteractionUC)

    bz            =   BZ([kSize, kSize])
    FillBZ!(bz, HoppingUC)

    bdgH           =   Hamiltonian(HoppingUC, PairingUC, bz)
    DiagonalizeHamiltonian!(bdgH)

    bdgModel    =   BdGModel(HoppingUC, PairingUC, bz, bdgH ; T=T, filling=filling, stat=stat)
    SolveModel!(bdgModel)

    bdgmft         =   BdGMFT(bdgModel, HoppingBlock, PairingBlock, InteractionBlock, InterQuarticToHopping, InterQuarticToPairing)
    fileName    =   "./Sample/SquaretJ/filling=$(round(filling, digits=3))_t1=$(round(t1, digits=3))_J=$(round(J, digits=3))_wtWeiss.jld2"
    SolveMFT!(bdgmft, fileName)

    println(p_dx2y2.value[end])
end