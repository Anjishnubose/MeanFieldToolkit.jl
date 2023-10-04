using TightBindingToolkit, MeanFieldToolkit, LinearAlgebra

############# Dual honeycomb lattice
a1   =  [ 3/2, sqrt(3)/2]
a2   =  [-3/2, sqrt(3)/2]
##### 6 sublattices
b1   =  [ 0.0 , 0.0 ]
b2   =  [ 1/2 , -1/(2 * sqrt(3)) ]
b3   =  [ 1/2 , -sqrt(3)/(2) ]
b4   =  [ 0.0 , -2/sqrt(3)   ]
b5   =  [-1/2 , -sqrt(3)/(2) ]
b6   =  [-1/2 , -1/(2 * sqrt(3))]

firstNNdistance     = 1.0/sqrt(3)
secondNNdistance    = 1.0
thirdNNdistance     = 2/sqrt(3)

UC      =   UnitCell([a1, a2], 2, 2)
UC.BC   =   [0.0, 0.0]

AddBasisSite!.(Ref(UC), [b1, b2, b3, b4, b5, b6])

SpinVec     =   SpinMats(1//2)
##### XY Spin Model exchange matrix
Jmatrix     =   [1.0 0.0 0.0 ; 0.0 1.0 0.0 ; 0.0 0.0 0.0]
##### XY Spin interaction written in terms of 4-parton interactions --> rank-4 array.
U           =   SpinToPartonCoupling(Jmatrix, 1//2)

JIntraParam     =   Param(1.0,  4)
AddAnisotropicBond!(JIntraParam, UC, 1, 2, [ 0, 0], U, firstNNdistance, "Intra Spin Interaction")
AddAnisotropicBond!(JIntraParam, UC, 2, 3, [ 0, 0], U, firstNNdistance, "Intra Spin Interaction")
AddAnisotropicBond!(JIntraParam, UC, 3, 4, [ 0, 0], U, firstNNdistance, "Intra Spin Interaction")
AddAnisotropicBond!(JIntraParam, UC, 4, 5, [ 0, 0], U, firstNNdistance, "Intra Spin Interaction")
AddAnisotropicBond!(JIntraParam, UC, 5, 6, [ 0, 0], U, firstNNdistance, "Intra Spin Interaction")
AddAnisotropicBond!(JIntraParam, UC, 6, 1, [ 0, 0], U, firstNNdistance, "Intra Spin Interaction")

JInterParam     =   Param(1.0, 4)
AddAnisotropicBond!(JInterParam, UC, 1, 4, [ 1,  1], U, firstNNdistance, "Inter Spin Interaction")
AddAnisotropicBond!(JInterParam, UC, 3, 6, [ 0, -1], U, firstNNdistance, "Inter Spin Interaction")
AddAnisotropicBond!(JInterParam, UC, 5, 2, [-1,  0], U, firstNNdistance, "Inter Spin Interaction")

Interactions   =   [JIntraParam, JInterParam]

JInters        =   collect(range(-3.0, 0.0, 16))
##### Brillouin zone
const n       =   10
const kSize   =   6 * n + 3
bz            =   BZ([kSize, kSize])
FillBZ!(bz, UC)

##### Thermodynamic parameters
const T         =   0.001
const filling   =   0.5
const stat      =   -1

##### Mixing alpha used for self-consistency solver
const mixingAlpha    =    0.5

for JInter in JInters

    push!(JInterParam.value, JInter)


    ##### Order parameters
    tIntraParam     =   Param(1.0, 2)
    AddAnisotropicBond!(tIntraParam, UC, 1, 2, [ 0, 0], SpinVec[4], firstNNdistance, "Intra hopping")
    AddAnisotropicBond!(tIntraParam, UC, 2, 3, [ 0, 0], SpinVec[4], firstNNdistance, "Intra hopping")
    AddAnisotropicBond!(tIntraParam, UC, 3, 4, [ 0, 0], SpinVec[4], firstNNdistance, "Intra hopping")
    AddAnisotropicBond!(tIntraParam, UC, 4, 5, [ 0, 0], SpinVec[4], firstNNdistance, "Intra hopping")
    AddAnisotropicBond!(tIntraParam, UC, 5, 6, [ 0, 0], SpinVec[4], firstNNdistance, "Intra hopping")
    AddAnisotropicBond!(tIntraParam, UC, 6, 1, [ 0, 0], SpinVec[4], firstNNdistance, "Intra hopping")

    tInterParam     =   Param(1.0, 2)
    AddAnisotropicBond!(tInterParam, UC, 1, 4, [ 1,  1], SpinVec[4], firstNNdistance, "Inter hopping")
    AddAnisotropicBond!(tInterParam, UC, 3, 6, [ 0, -1], SpinVec[4], firstNNdistance, "Inter hopping")
    AddAnisotropicBond!(tInterParam, UC, 5, 2, [-1,  0], SpinVec[4], firstNNdistance, "Inter hopping")

    tIntraSzParam     =   Param(1.0, 2)
    AddAnisotropicBond!(tIntraSzParam, UC, 1, 2, [ 0, 0], SpinVec[3], firstNNdistance, "Intra hopping Sz")
    AddAnisotropicBond!(tIntraSzParam, UC, 2, 3, [ 0, 0], SpinVec[3], firstNNdistance, "Intra hopping Sz")
    AddAnisotropicBond!(tIntraSzParam, UC, 3, 4, [ 0, 0], SpinVec[3], firstNNdistance, "Intra hopping Sz")
    AddAnisotropicBond!(tIntraSzParam, UC, 4, 5, [ 0, 0], SpinVec[3], firstNNdistance, "Intra hopping Sz")
    AddAnisotropicBond!(tIntraSzParam, UC, 5, 6, [ 0, 0], SpinVec[3], firstNNdistance, "Intra hopping Sz")
    AddAnisotropicBond!(tIntraSzParam, UC, 6, 1, [ 0, 0], SpinVec[3], firstNNdistance, "Intra hopping Sz")

    tInterSzParam     =   Param(1.0, 2)
    AddAnisotropicBond!(tInterSzParam, UC, 1, 4, [ 1,  1], SpinVec[3], firstNNdistance, "Inter hopping Sz")
    AddAnisotropicBond!(tInterSzParam, UC, 3, 6, [ 0, -1], SpinVec[3], firstNNdistance, "Inter hopping Sz")
    AddAnisotropicBond!(tInterSzParam, UC, 5, 2, [-1,  0], SpinVec[3], firstNNdistance, "Inter hopping Sz")

    HoppingOrders   =   [tIntraParam, tInterParam, tIntraSzParam, tInterSzParam]

    H             =   Hamiltonian(UC, bz)
    DiagonalizeHamiltonian!(H)
    M             =   Model(UC, bz, H ; T=T, filling=filling, stat=stat)
    ##### Build the mean-field theory object, with chosen scaling such that on-site ordering is suppressed.
    mft           =   TightBindingMFT(M, HoppingOrders, Interactions, InterQuarticToHopping, Dict{String, Float64}("ij" => 1.0, "ii" => 0.0, "jj" => 0.0) )
    fileName      =   "./Sample/DualHoneycomb_Data/JInter=$(round(JInter, digits=2))_JIntra=$(round(JIntraParam.value[end], digits=2)).jld2"
    SolveMFT!(mft, fileName)

end
