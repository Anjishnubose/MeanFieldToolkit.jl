module MFTIter
    export DecomposeGr, MFTIterator

    using TightBindingToolkit, LinearAlgebra, Statistics

    using ..MeanFieldToolkit.MFTBonds: GetBondCoorelation
    using ..MeanFieldToolkit.Blocks: ParamBlock, UpdateBlock!
    using ..MeanFieldToolkit.TBMFT: TightBindingMFT, GetMFTEnergy
    using ..MeanFieldToolkit.BDGMFT: BdGMFT, GetMFTEnergy
    using ..MeanFieldToolkit.Build: BuildFromInteractions!

    ##### TODO : Try to vectorize maybe?
    function DecomposeGr(Gr::Array{Matrix{ComplexF64}, T}, param::Param{2}, uc::UnitCell{T}, bz::BZ) :: Float64 where {T}

        strengths   =   Float64[] 

        for bond in param.unitBonds
            ##### TODO : the extra - sign in offset is because right now G[r] = <f^{dagger}_0 . f_{-r}> ===> NEED TO FIX THIS

            G           =   GetBondCoorelation(Gr, bond, uc, bz)

            strength    =   real(tr( adjoint(bond.mat) * G) / (tr(adjoint(bond.mat) * bond.mat)))
            push!(strengths, strength)
        end

        return mean(strengths)

    end


    function MFTIterator(Strengths::Vector{Float64}, tbMFT::TightBindingMFT{T} ; OnSiteMatrices::Vector{Matrix{ComplexF64}} = SpinMats((tbMFT.TightBindingModel.uc.localDim-1)//2)) :: Vector{Float64} where {T}

        push!.( getproperty.(tbMFT.HoppingBlock.params, :value) , Strengths)
        UpdateBlock!(tbMFT.HoppingBlock)

        BuildFromInteractions!(tbMFT)

        H       =   Hamiltonian(tbMFT.TightBindingModel.uc, tbMFT.TightBindingModel.bz ; OnSiteMatrices = OnSiteMatrices)
        DiagonalizeHamiltonian!(H)

        tbMFT.TightBindingModel.Ham     =   H
        SolveModel!(tbMFT.TightBindingModel)

        push!(tbMFT.MFTEnergy, GetMFTEnergy(tbMFT))

        NewStrengths    =   DecomposeGr.(Ref(tbMFT.TightBindingModel.Gr), tbMFT.HoppingBlock.params, Ref(tbMFT.TightBindingModel.uc), Ref(tbMFT.TightBindingModel.bz))

        return NewStrengths
    end

    function MFTIterator(Strengths::Vector{Float64}, bdgMFT::BdGMFT{T} ; OnSiteMatrices::Vector{Matrix{ComplexF64}} = SpinMats((tbMFT.TightBindingModel.uc.localDim-1)//2)) :: Vector{Float64} where {T}

        push!.( getproperty.(bdgMFT.HoppingBlock.params, :value) , Strengths[begin : length(bdgMFT.HoppingBlock.params)])
        UpdateBlock!(bdgMFT.HoppingBlock)

        push!.( getproperty.(bdgMFT.PairingBlock.params, :value) , Strengths[length(bdgMFT.HoppingBlock.params) + 1 : end])
        UpdateBlock!(bdgMFT.PairingBlock)

        BuildFromInteractions!(bdgMFT)

        H       =   Hamiltonian(bdgMFT.bdgModel.uc_hop, bdgMFT.bdgModel.uc_pair, bdgMFT.bdgModel.bz ; OnSiteMatrices = OnSiteMatrices)
        DiagonalizeHamiltonian!(H)

        bdgMFT.bdgModel.Ham     =   H
        SolveModel!(bdgMFT.bdgModel)

        push!(bdgMFT.MFTEnergy, GetMFTEnergy(bdgMFT))

        NewStrengths    =   DecomposeGr.(Ref(bdgMFT.bdgModel.Gr), bdgMFT.HoppingBlock.params, Ref(bdgMFT.bdgModel.uc_hop), Ref(bdgMFT.bdgModel.bz))
        append!(NewStrengths, DecomposeGr.(Ref(bdgMFT.bdgModel.Fr), bdgMFT.PairingBlock.params, Ref(bdgMFT.bdgModel.uc_pair), Ref(bdgMFT.bdgModel.bz)))

        return NewStrengths
    end








end