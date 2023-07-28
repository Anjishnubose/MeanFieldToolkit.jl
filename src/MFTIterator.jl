module MFTIter
    export DecomposeGr, MFTIterator

    using TightBindingToolkit, LinearAlgebra, Statistics

    using ..MeanFieldToolkit.Blocks: ParamBlock, UpdateBlock!
    using ..MeanFieldToolkit.TBMFT: TightBindingMFT, GetMFTEnergy
    using ..MeanFieldToolkit.BDGMFT: BdGMFT, GetMFTEnergy
    using ..MeanFieldToolkit.Build: BuildFromInteractions!

    ##### TODO : Try to vectorize maybe?
    function DecomposeGr(Gr::Array{Matrix{ComplexF64}, T}, param::Param{2}, uc::UnitCell{T}, bz::BZ) :: Float64 where {T}

        strengths   =   Float64[] 

        for bond in param.unitBonds

            index       =   mod.((-bond.offset) , bz.gridSize) .+ ones(Int64, length(bond.offset)) 
            ##### TODO : the extra - sign in offset is because right now G[r] = <f^{dagger}_0 . f_{-r}> ===> NEED TO FIX THIS
            b1          =   uc.localDim * (bond.base   - 1) + 1
            b2          =   uc.localDim * (bond.target - 1) + 1

            G           =   Gr[index...][b1 : b1 + uc.localDim - 1, b2 : b2 + uc.localDim - 1]

            strength    =   real(tr( adjoint(bond.mat) * G) / (tr(adjoint(bond.mat) * bond.mat)))
            push!(strengths, strength)
        end

        return mean(strengths)

    end


    function MFTIterator(Strengths::Vector{Float64}, tbMFT::TightBindingMFT{T}) :: Vector{Float64} where {T}

        push!.( getproperty.(tbMFT.HoppingBlock.params, :value) , Strengths)
        UpdateBlock!(tbMFT.HoppingBlock)

        BuildFromInteractions!(tbMFT)
        push!(tbMFT.MFTEnergy, GetMFTEnergy(tbMFT))

        H       =   Hamiltonian(tbMFT.TightBindingModel.uc, tbMFT.TightBindingModel.bz)
        DiagonalizeHamiltonian!(H)

        tbMFT.TightBindingModel.Ham     =   H
        SolveModel!(tbMFT.TightBindingModel)

        NewStrengths    =   DecomposeGr.(Ref(tbMFT.TightBindingModel.Gr), tbMFT.HoppingBlock.params, Ref(tbMFT.TightBindingModel.uc), Ref(tbMFT.TightBindingModel.bz))

        return NewStrengths
    end

    function MFTIterator(Strengths::Vector{Float64}, bdgMFT::BdGMFT{T}) :: Vector{Float64} where {T}

        push!.( getproperty.(bdgMFT.HoppingBlock.params, :value) , Strengths[begin : length(bdgMFT.HoppingBlock.params)])
        UpdateBlock!(bdgMFT.HoppingBlock)

        push!.( getproperty.(bdgMFT.PairingBlock.params, :value) , Strengths[length(bdgMFT.HoppingBlock.params) + 1 : end])
        UpdateBlock!(bdgMFT.PairingBlock)

        BuildFromInteractions!(bdgMFT)
        push!(bdgMFT.MFTEnergy, GetMFTEnergy(bdgMFT))

        H       =   Hamiltonian(bdgMFT.bdgModel.uc_hop, bdgMFT.bdgModel.uc_pair, bdgMFT.bdgModel.bz)
        DiagonalizeHamiltonian!(H)

        bdgMFT.bdgModel.Ham     =   H
        SolveModel!(bdgMFT.bdgModel)

        NewStrengths    =   DecomposeGr.(Ref(bdgMFT.bdgModel.Gr), bdgMFT.HoppingBlock.params, Ref(bdgMFT.bdgModel.uc_hop), Ref(bdgMFT.bdgModel.bz))
        append!(NewStrengths, DecomposeGr.(Ref(bdgMFT.bdgModel.Fr), bdgMFT.PairingBlock.params, Ref(bdgMFT.bdgModel.uc_pair), Ref(bdgMFT.bdgModel.bz)))

        return NewStrengths
    end








end