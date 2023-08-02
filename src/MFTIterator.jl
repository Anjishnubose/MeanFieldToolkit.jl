module MFTIter
    export DecomposeGr, MFTIterator

    using TightBindingToolkit, LinearAlgebra, Statistics

    using ..MeanFieldToolkit.MFTBonds: GetBondCoorelation
    using ..MeanFieldToolkit.Blocks: ParamBlock, UpdateBlock!
    using ..MeanFieldToolkit.TBMFT: TightBindingMFT, GetMFTEnergy
    using ..MeanFieldToolkit.BDGMFT: BdGMFT, GetMFTEnergy
    using ..MeanFieldToolkit.Build: BuildFromInteractions!

    #####/// TODO : Try to vectorize maybe? Literally no need
    function DecomposeGr(Gr::Array{Matrix{ComplexF64}, T}, param::Param{2, R}, uc::UnitCell{T}, bz::BZ) :: R where {T, R <: Union{Float64, ComplexF64}}

        strengths   =   R[] 

        for bond in param.unitBonds

            G           =   GetBondCoorelation(Gr, bond, uc, bz)

            decomposition   =   (tr( adjoint(bond.mat) * G) / (tr(adjoint(bond.mat) * bond.mat)))
            strength        =   (R == Float64 ? real(decomposition) : decomposition)

            push!(strengths, strength)
        end

        return mean(strengths)
    end


    function MFTIterator(Strengths::Vector{R}, tbMFT::TightBindingMFT{T, R}) :: Vector{R} where {T, R <: Union{Float64, ComplexF64}}

        push!.( getproperty.(tbMFT.HoppingBlock.params, :value) , Strengths)
        UpdateBlock!(tbMFT.HoppingBlock)

        BuildFromInteractions!(tbMFT)

        H       =   Hamiltonian(tbMFT.TightBindingModel.uc, tbMFT.TightBindingModel.bz)
        DiagonalizeHamiltonian!(H)

        tbMFT.TightBindingModel.Ham     =   H
        SolveModel!(tbMFT.TightBindingModel)

        push!(tbMFT.MFTEnergy, GetMFTEnergy(tbMFT))

        NewStrengths    =   DecomposeGr.(Ref(tbMFT.TightBindingModel.Gr), tbMFT.HoppingBlock.params, Ref(tbMFT.TightBindingModel.uc), Ref(tbMFT.TightBindingModel.bz))

        return NewStrengths
    end


    function MFTIterator(Strengths::Vector{R}, bdgMFT::BdGMFT{T, R, R}) :: Vector{Float64} where {T, R <: Union{Float64, ComplexF64}}

        push!.( getproperty.(bdgMFT.HoppingBlock.params, :value) , Strengths[begin : length(bdgMFT.HoppingBlock.params)])
        UpdateBlock!(bdgMFT.HoppingBlock)

        push!.( getproperty.(bdgMFT.PairingBlock.params, :value) , Strengths[length(bdgMFT.HoppingBlock.params) + 1 : end])
        UpdateBlock!(bdgMFT.PairingBlock)

        BuildFromInteractions!(bdgMFT)

        H       =   Hamiltonian(bdgMFT.bdgModel.uc_hop, bdgMFT.bdgModel.uc_pair, bdgMFT.bdgModel.bz)
        DiagonalizeHamiltonian!(H)

        bdgMFT.bdgModel.Ham     =   H
        SolveModel!(bdgMFT.bdgModel)

        push!(bdgMFT.MFTEnergy, GetMFTEnergy(bdgMFT))

        NewStrengths    =   DecomposeGr.(Ref(bdgMFT.bdgModel.Gr), bdgMFT.HoppingBlock.params, Ref(bdgMFT.bdgModel.uc_hop), Ref(bdgMFT.bdgModel.bz))
        append!(NewStrengths, DecomposeGr.(Ref(bdgMFT.bdgModel.Fr), bdgMFT.PairingBlock.params, Ref(bdgMFT.bdgModel.uc_pair), Ref(bdgMFT.bdgModel.bz)))

        return NewStrengths
    end








end