using TightBindingToolkit, MeanFieldToolkit, FixedPointToolkit, LinearAlgebra, Plots, LaTeXStrings

JInters        =   collect(range(-2.0, 0.0, 41))[16:24]

energies       =   []
Orders         =   []

for JInter in JInters

    fileName      =   "./Sample/DualHoneycomb_Data/JInter=$(round(JInter, digits=2))_JIntra=$(round(-1.0, digits=2))_Flipped.jld2"
    data    =   ReadMFT(fileName)
    push!(Orders, data["Expectations"])
    push!(energies, data["MFT"].MFTEnergy[end])

    println(data["Convergence"])

    GC.gc()
end