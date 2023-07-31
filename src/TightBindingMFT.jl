module TBMFT

    export TightBindingMFT, GetMFTEnergy

    using TightBindingToolkit, LinearAlgebra, Logging

    using ..MeanFieldToolkit.MFTBonds: GetBondCoorelation
    using ..MeanFieldToolkit.Blocks: ParamBlock

    mutable struct TightBindingMFT{T} 
        ##### TightBinding Model for MFT, and the ParamBlock containing the expected order parameters of MFT
        TightBindingModel   ::  Model
        HoppingBlock        ::  ParamBlock{2}
        ##### The ParamBlock tracking information about the Interacting UnitCell, and the corresponding MFT decomposition functions
        InteractionBlock    ::  Vector{ParamBlock{T}}
        MFTDecomposition    ::  Vector{Function}
        ##### The MFT expectation value of the full interacting Hamiltonian
        MFTEnergy           ::  Vector{Float64}
        ##### The relative scaling b/w different MFT channels
        MFTScaling          ::  Dict{String, Float64}
        ##### The user defined labels of the different MFT channels
        ChannelLabels       ::  Dict{String, String}


        function TightBindingMFT(TightBindingModel::Model, HoppingBlock::ParamBlock{2}, InteractionBlock::Vector{ParamBlock{T}} , MFTDecomposition::Vector{Function} ; ChannelLabels :: Dict{String, String} = Dict{String, String}("ij" => "Hopping", "ii" => "Hopping On-Site", "jj" => "Hopping On-Site")) where {T}

            @warn "`MFTScaling` attribute not passed. Resorting to default values of uniform relative scaling for every channel!"
            MFTScaling      =   Dict{String, Float64}("ij" => 1.0, "ii" => 1.0, "jj" => 1.0) 

            @assert IsSameUnitCell(TightBindingModel.uc, HoppingBlock.uc) "Inconsistency between Tight-Binding Unit Cell and Expectation Unit Cell"

            return new{T}(TightBindingModel, HoppingBlock, InteractionBlock, MFTDecomposition, Float64[], MFTScaling, ChannelLabels)
        end

        function TightBindingMFT(TightBindingModel::Model, HoppingBlock::ParamBlock{2}, InteractionBlock::Vector{ParamBlock{T}}, MFTDecomposition::Vector{Function}, MFTScaling::Dict{String, Float64} ; ChannelLabels :: Dict{String, String} = Dict{String, String}("ij" => "Hopping", "ii" => "Hopping On-Site", "jj" => "Hopping On-Site")) where {T}

            @assert IsSameUnitCell(TightBindingModel.uc, HoppingBlock.uc) "Inconsistency between Tight-Binding Unit Cell and Expectation Unit Cell"

            return new{T}(TightBindingModel, HoppingBlock, InteractionBlock, MFTDecomposition, Float64[], MFTScaling, ChannelLabels)
        end

        function TightBindingMFT(TightBindingModel::Model, HoppingBlock::ParamBlock{2}, InteractionBlock::ParamBlock{T} , MFTDecomposition::Function ; ChannelLabels :: Dict{String, String} = Dict{String, String}("ij" => "Hopping", "ii" => "Hopping On-Site", "jj" => "Hopping On-Site")) where {T}

            return TightBindingMFT(TightBindingModel, HoppingBlock, ParamBlock{T}[InteractionBlock], Function[MFTDecomposition] ; ChannelLabels = ChannelLabels)
        end

        function TightBindingMFT(TightBindingModel::Model, HoppingBlock::ParamBlock{2}, InteractionBlock::ParamBlock{T}, MFTDecomposition::Function, MFTScaling::Dict{String, Float64} ; ChannelLabels :: Dict{String, String} = Dict{String, String}("ij" => "Hopping", "ii" => "Hopping On-Site", "jj" => "Hopping On-Site")) where {T}

            return TightBindingMFT(TightBindingModel, HoppingBlock, ParamBlock{T}[InteractionBlock], Function[MFTDecomposition], MFTScaling; ChannelLabels = ChannelLabels)
        end

    end

    ##### /// TODO: Add Free Hopping energies also 
    ##### /// TODO: Test!!!!
    function GetMFTEnergy(tbMFT::TightBindingMFT{T}) :: Float64 where {T}

        Energy      =   0.0
        lookup      =   Lookup(tbMFT.TightBindingModel.uc)

        for BondKey in keys(lookup)

            G_ij        =   GetBondCoorelation(tbMFT.TightBindingModel.Gr, BondKey..., tbMFT.TightBindingModel.uc, tbMFT.TightBindingModel.bz)

            t_ij        =   lookup[BondKey]
            Energy      +=  sum((t_ij .* G_ij))
        end

        return real(Energy) / length(tbMFT.HoppingBlock.uc.basis)
    end










end