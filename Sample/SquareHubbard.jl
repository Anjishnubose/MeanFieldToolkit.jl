using MeanFieldToolkit, TightBindingToolkit, FixedPointToolkit
using LinearAlgebra, Distributions, Base.Threads
##### YOU NEED TO CREATE THE FOLDER Sample/SquareHubbard_Data BEFORE RUNNING THIS SCRIPT
########## Defining square lattice 
##### Primitives
const a1 = [0.0, 1.0]
const a2 = [1.0, 0.0]

const b1 = [0.0, 0.0]

const InitialField = zeros(Float64, 4)

const n = 10
const kSize = 6 * n + 3

SpinVec = SpinMats(1 // 2)
const t = 1.0
const U = 4.0

##### Thermodynamic parameters
const T = 0.001
const stat = -1
const mixingAlpha = 0.5


fillings = collect(LinRange(0.1, 0.5, 20))
UC = UnitCell([a1, a2], 2, 2)

AddBasisSite!(UC, b1)
##### HoppingParams
t1 = -t
t1Param = Param(t1, 2)
AddIsotropicBonds!(t1Param, UC, 1.0, SpinVec[4], "t1")

HoppingParams = [t1Param]
n_up = [1.0 0.0; 0.0 0.0]
n_down = [0.0 0.0; 0.0 1.0]
Hubbard = DensityToPartonCoupling(n_up, n_down)

UParam = Param(U, 4)
AddIsotropicBonds!(UParam, UC, 0.0, Hubbard, "Hubbard Interaction")
CreateUnitCell!(UC, HoppingParams)

bz = BZ([kSize, kSize])
FillBZ!(bz, UC)
for filling in fillings
    ##### Hopping expectation params
    t_s = Param(1.0, 2)
    AddIsotropicBonds!(t_s, UC, 1.0, SpinVec[4], "s Hopping")
    Neel = Param(1.0, 2)
    AddAnisotropicBond!(Neel, UC, 1, 1, [0, 0], SpinVec[3], 0.0, "Neel order")
    AddAnisotropicBond!(Neel, UC, 2, 2, [0, 0], -SpinVec[3], 0.0, "Neel order")

    ChiParams = [t_s, Neel]

    ####################### Interaction Block                                  


    H = Hamiltonian(UC, bz)
    DiagonalizeHamiltonian!(H)

    M = Model(UC, bz, H; T=T, filling=filling, stat=stat)
    SolveModel!(M)
    mft = TightBindingMFT(M, ChiParams, [UParam], IntraQuarticToHopping)
    fileName = "./SquareHubbard/filling=$(round(filling, digits=3))_U=$(round(U, digits=2))_t1=$(round(t1, digits=2)).jld2"
    SolveMFT!(mft, fileName)

end
