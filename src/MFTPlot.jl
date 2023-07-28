module MFTPlot
    export PlotMFT!, PlotMFTEnergy!

    using Plots

    using ..MeanFieldToolkit.Blocks: ParamBlock
    using ..MeanFieldToolkit.TBMFT: TightBindingMFT
    using ..MeanFieldToolkit.BDGMFT: BdGMFT


    function PlotMFT!(mft::TightBindingMFT ; plot_labels::Vector{String} = getproperty.(mft.HoppingBlock.params, :label), plot_legend::Bool = true)

        p = plot(grid=false, legend = plot_legend, bg_legend = :transparent)

        for param in mft.HoppingBlock.params
            if param.label in plot_labels

                plot!(param.value, marker = :circle, lw = 2.0, label = param.label)
            end
        end

        xlabel!("Iterations", guidefontsize = 9)
        ylabel!("Parameters", guidefontsize = 9)
        title!("MFT parameters", titlefontsize = 12)

        return p
    end


    function PlotMFT!(mft::BdGMFT ; plot_labels::Vector{String} = getproperty.(mft.HoppingBlock.params, :label), plot_legend::Bool = true)

        p = plot(grid=false, legend = plot_legend, bg_legend = :transparent)

        for param in mft.HoppingBlock.params
            if param.label in plot_labels

                plot!(param.value, marker = :circle, lw = 2.0, label = param.label)
            end
        end

        for param in mft.PairingBlock.params
            if param.label in plot_labels

                plot!(param.value, marker = :square, lw = 2.0, label = param.label)
            end
        end

        xlabel!("Iterations", guidefontsize = 9)
        ylabel!("Parameters", guidefontsize = 9)
        title!("BdG MFT parameters", titlefontsize = 12)

        return p
    end


    function PlotMFTEnergy!(mft::T) where {T<:Union{TightBindingMFT, BdGMFT}}

        plot!(mft.MFTEnergy, marker = :circle, lw = 2.0, label = "MFT Energy")

        xlabel!("Iterations", guidefontsize = 9)
        ylabel!("Energy", guidefontsize = 9)

        return p
    end





































end