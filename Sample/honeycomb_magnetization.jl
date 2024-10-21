include("../src/MeanFieldToolkit.jl")
using .MeanFieldToolkit
using TightBindingToolkit, FixedPointToolkit, JLD2, Plots, LaTeXStrings

##### primitive vectors
const a1  =   [1/2, sqrt(3)/2]
const a2  =   [-1/2, sqrt(3)/2]

const SpinVec = SpinMats(1//2)
##### sublattices
const b1  =   [0.0, 0.0]
const b2  =   [0.0, 1/sqrt(3)]

function honeycomb(t1::Float64, t3::Float64, inPlaneField::Float64, outPlaneField::Float64)
    HoppingUC       =   UnitCell([a1, a2], 2, 2)
    AddBasisSite!.(Ref(HoppingUC), [b1, b2])

    tNN = Param(t1, 2)
    t3NN = Param(t3, 2)
    InPlaneField = Param(inPlaneField, 2)
    OutPlaneField = Param(outPlaneField, 2)

    AddIsotropicBonds!(tNN, HoppingUC, 1/sqrt(3), 2*SpinVec[3], "NN Hopping")
    AddIsotropicBonds!(t3NN, HoppingUC, 2/sqrt(3), 2*SpinVec[3], "3NN Hopping")
    AddIsotropicBonds!(InPlaneField, HoppingUC, 0.0, SpinVec[1], "InPlane Field")
    AddIsotropicBonds!(OutPlaneField, HoppingUC, 0.0, SpinVec[3], "OutPlane Field")

    CreateUnitCell!(HoppingUC,[tNN,t3NN,InPlaneField,OutPlaneField])

    return HoppingUC
end

##### Thermodynamic parameters
const T         =   0.05
const stat      =   -1
const filling   = 0.5

function honeycombMFT(t1::Float64, t3::Float64, inPlaneField::Float64, outPlaneField::Float64,
                        J1::Float64, J3::Float64,
                        fileName::String,
                        scalings::Dict{String, Any} = Dict{String, Any}("ij" => 1.0, "ii" => 1.0, "jj" => 1.0))
    HoppingUC = honeycomb(t1, t3, inPlaneField, outPlaneField)
    bz = BZ([33, 33])
    FillBZ!(bz, HoppingUC)

    Jmatrix     =   [[1.0 0.0 0.0];[0.0 1.0 0.0];[0.0 0.0 0.0]]
    U           =   SpinToPartonCoupling(Jmatrix, 1//2)
    JParam3      =   Param(J3, 4)
    JParam1      =   Param(J1, 4)
    AddIsotropicBonds!(JParam3,HoppingUC, 2/sqrt(3), U, "J3 Interaction")
    AddIsotropicBonds!(JParam1,HoppingUC, 1/sqrt(3), U, "J1 Interaction")
    InteractionParams   =  [JParam3,JParam1]

    ##### Order parameters
    ferro = Param(1.0, 2)
    direction = 1
    ##### Ferromagnetic order
    ferroSigns = [1, 1]
    for (b, basis) in enumerate(HoppingUC.basis)
        AddAnisotropicBond!(ferro, HoppingUC, b, b, [0, 0], ferroSigns[b] * SpinVec[direction], 0.0, "ferro order along $(direction)")
    end

    t1Chi = Param(1.0, 2)
    AddIsotropicBonds!(t1Chi, HoppingUC, 1/sqrt(3), 2*SpinVec[3], "t1 expectation")
    t3Chi = Param(1.0, 2)
    AddIsotropicBonds!(t3Chi, HoppingUC, 2/sqrt(3), 2*SpinVec[3], "t3 expectation")

    expectations = [ferro, t1Chi, t3Chi]

    H           =   Hamiltonian(HoppingUC, bz)
    DiagonalizeHamiltonian!(H)

    model    =   Model(HoppingUC, bz, H ; T=T, filling=filling, stat=stat)
    mft      =   TightBindingMFT(model, expectations, InteractionParams, InterQuarticToHopping, scalings)

    if fileName != ""
        sc = SolveMFT!(mft, fileName; max_iter=200, tol=1e-4);
    else
        sc = SolveMFT!(mft; max_iter=200, tol=1e-4)
    end
    # save(fileName, Dict("ferro"=> ferro.value[end], "t1" => t1Chi.value[end], "t3" => t3Chi.value[end],
    #     "energy" => mft.MFTEnergy[end]))

    return mft

end

input = load("./J1=-1.0_J3=0.3_T=0.05_wBx_Scaling=0.0.jld2")


##################### parameters
const J = 1.0
const g = 1.0
const alpha1 = 0.75
const alpha3 = 0.7*alpha1

# const t1as = -0.5 * J * alpha1 * abs.(input["t1s"]) * g
# const t3as = 0.2 * t1as
const t1 = 0.0
const t3 = 0.0
const outPlaneField = 0.0

# J = 3.0
# theta = 0.8*pi
# const J1 = J*cos(theta)
# const J3 = J*sin(theta)

# const J1 = -(1-alpha1)*J * g
# const J3 = (1-alpha3)*J*0.25 * g
const J1 = -1.0*J*g
const J3 = 0.3*J*g

const inPlanes = collect(range(0.0, 0.5, length=51))*J
# const Bx = -1.0
# const theta = 1.0*pi
# const Js = collect(LinRange(0.0, 2.0, 41))

ferros = Float64[]
t1s = Float64[]
t3s = Float64[]
energies = Float64[]

# alpha1 = 1.0
# OnSiteScaling = 1-alpha1
# scalings = Dict{String, Float64}("ij" => alpha1, "ii" => OnSiteScaling, "jj" => OnSiteScaling)
scalings1 = Dict{String, Float64}("ij" => alpha1, "ii" => 1.0 - alpha1, "jj" => 1.0 - alpha1)
scalings3 = Dict{String, Float64}("ij" => alpha3, "ii" => 1.0 - alpha3, "jj" => 1.0 - alpha3)
scalings = Dict{String, Any}("J1 Interaction" => scalings1, "J3 Interaction" => scalings3)

# for J in Js
#     J1 = J*cos(theta)
#     J3 = J*sin(theta)
#     fileName = "./MeanFieldToolkit.jl/Sample/HoneycombXXZ_Data/J1=$(round(J1, digits=2))_J3=$(round(J3, digits=2))_Bx=$(Bx)_T=$(T)_wBx_Scaling=$(OnSiteScaling).jld2"
#     sc = honeycombMFT(t1, t3, Bx, outPlaneField, J1, J3, fileName, scalings)
#     push!(ferros, sc.HoppingOrders[1].value[end])
#     push!(t1s, sc.HoppingOrders[2].value[end])
#     push!(t3s, sc.HoppingOrders[3].value[end])
#     push!(energies, sc.MFTEnergy[end])
#     println("J = $(J) done")
# end

for (b, Bx) in enumerate(inPlanes)
    # fileName = "./Sample/HoneycombXXZ/J1=$(round(J1, digits=2))_J3=$(round(J3, digits=2))_Bx=$(Bx)_T=$(T)_kSpace.jld2"
    fileName = ""
    sc = honeycombMFT(0.0, 0.0, Bx, outPlaneField, J1, J3, fileName, scalings)
    push!(ferros, sc.HoppingOrders[1].value[end])
    push!(t1s, sc.HoppingOrders[2].value[end])
    push!(t3s, sc.HoppingOrders[3].value[end])
    push!(energies, sc.MFTEnergy[end])
    println("Bx = $(Bx) done")
end
# save("./J1=$(round(J1, digits=2))_J3=$(round(J3, digits=2))_T=$(T)_wBx_Scaling=$(round(alpha1, digits=2))_ratio=0.5.jld2",
#         Dict("ferros" => ferros, "energies" => energies, "inPlanes" => inPlanes,
#             "t1s" => t1s, "t3s" => t3s))

# save("./J1=$(round(J1, digits=2))_J3=$(round(J3, digits=2))_T=$(T)_wBx_Scaling=$(OnSiteScaling).jld2",
#     Dict("ferros" => ferros, "energies" => energies, "inPlanes" => inPlanes,
#     "t1s" => t1s, "t3s" => t3s))

# save("./Sample/HoneycombXXZ/Dirac_J1=$(round(J1, digits=2))_J3=$(round(J3, digits=2))_T=$(T)_wBx_Scaling=$(OnSiteScaling).jld2",
#         Dict("ferros" => ferros, "energies" => energies, "inPlanes" => inPlanes,
#             "t1s" => t1s, "t3s" => t3s))

# save("./Sample/HoneycombXXZ/Dirac_J=$(J)_theta=$(round(theta/pi, digits=2))pi_T=$(T)_wBx_Scaling=$(OnSiteScaling).jld2",
#         Dict("ferros" => ferros, "energies" => energies, "inPlanes" => inPlanes,
#                 "t1s" => t1s, "t3s" => t3s))

# Js = [0.0, 1.0, 2.0, 3.0]

# datas = []
# for J in Js
#     push!(datas, load("./Sample/HoneycombXXZ/Dirac_J=$(J)_theta=$(round(theta/pi, digits=2))pi_T=$(T)_wBx_Scaling=$(OnSiteScaling).jld2"))
# end

# p = plot(framestyle=:box, grid=false,
#         guidefont = "Computer Modern", legendfont = "Computer Modern", tickfont = "Computer Modern",
#         guidefontsize = 14, legendfontsize = 12, tickfontsize = 12,
#         xlabel = L"B/t_1",
#         ylabel = L"M_x",
#         title = L"\theta=0.8\pi")

# for (j, J) in enumerate(Js)
#     plot!(p, inPlanes, abs.(datas[j]["ferros"]), label = L"J=%$(J)", lw = 2)
# end



# scalings = [0.0, 0.1, 0.2, 0.3]
# datas = []

# for scaling in scalings
#     push!(datas, load("./Sample/HoneycombXXZ/J1=$(round(J1, digits=2))_J3=$(round(J3, digits=2))_T=$(T)_wBx_Scaling=$(scaling).jld2"))
# end

# pMx = plot(framestyle=:box, grid=false,
#         guidefont = "Computer Modern", legendfont = "Computer Modern", tickfont = "Computer Modern",
#         guidefontsize = 14, legendfontsize = 12, tickfontsize = 12,
#         xlabel = L"B/J_1",
#         ylabel = L"M_x",
#         xlims = (-0.025, 1.025),
#         title = L"J_1=-1.0\,,J_3=0.32")

# for (s, scaling) in enumerate(scalings)
#     plot!(pMx, inPlanes, abs.(datas[s]["ferros"]), label = L"\alpha=%$(1-scaling)", lw = 2, marker=:o)
# end


# pt1 = plot(framestyle=:box, grid=false,
#         guidefont = "Computer Modern", legendfont = "Computer Modern", tickfont = "Computer Modern",
#         guidefontsize = 14, legendfontsize = 12, tickfontsize = 12,
#         xlabel = L"B/J_1",
#         ylabel = L"t_1",
#         xlims = (-0.025, 1.025),
#         title = L"J_1=-1.0\,,J_3=0.32")

# for (s, scaling) in enumerate(scalings)
#     plot!(pt1, inPlanes, abs.(datas[s]["t1s"])/2, label = L"\alpha=%$(1-scaling)", lw = 2, marker=:o)
# end
