module TBMFT

    export TightBindingMFT, GetMFTEnergy

    using TightBindingToolkit, LinearAlgebra, Logging

    using ..MeanFieldToolkit.MFTBonds: GetBondCoorelation
    using ..MeanFieldToolkit.Blocks: ParamBlock


@doc """
`TightBindingMFT{T, R}` is a data type representing a general mean-field simulation on a tight-binding model.

# Attributes
- `TightBindingModel  ::  Model`: The tight-binding model on which mean-field simulations are going to run.
- `HoppingBlock       ::  ParamBlock{2, R}`: a block of order parameters to decompose the interactions in during MFT.
- `InteractionBlock   ::  ParamBlock{2, FLoat64}`: the block containing all the information of the interactions acting on the model.
- `MFTDecomposition   ::  Vector{Function}` : the decomposition function which describes how to take an interaction array + expectation values and give back tight-binding hoppings.
- `MFTScaling         ::  Dict{String, Float64}`: relative scaling parameters for different mean-field channels.
- `ChannelLabels      ::  Dict{String, String}`: The labels of the different mean-field channels.

Initialize this structure using 
```julia
TightBindingMFT(TightBindingModel::Model, HoppingBlock::ParamBlock{2, R}, InteractionBlock::Vector{ParamBlock{T, Float64}} , MFTDecomposition::Vector{Function} ; ChannelLabels :: Dict{String, String} = Dict{String, String}("ij" => "Hopping", "ii" => "Hopping On-Site", "jj" => "Hopping On-Site"))
TightBindingMFT(TightBindingModel::Model, HoppingBlock::ParamBlock{2, R}, InteractionBlock::Vector{ParamBlock{T, Float64}}, MFTDecomposition::Vector{Function}, MFTScaling::Dict{String, Float64} ; ChannelLabels :: Dict{String, String} = Dict{String, String}("ij" => "Hopping", "ii" => "Hopping On-Site", "jj" => "Hopping On-Site"))
TightBindingMFT(TightBindingModel::Model, HoppingBlock::ParamBlock{2, R}, InteractionBlock::ParamBlock{T, Float64} , MFTDecomposition::Function ; ChannelLabels :: Dict{String, String} = Dict{String, String}("ij" => "Hopping", "ii" => "Hopping On-Site", "jj" => "Hopping On-Site"))
TightBindingMFT(TightBindingModel::Model, HoppingBlock::ParamBlock{2, R}, InteractionBlock::ParamBlock{T, Float64}, MFTDecomposition::Function, MFTScaling::Dict{String, Float64} ; ChannelLabels :: Dict{String, String} = Dict{String, String}("ij" => "Hopping", "ii" => "Hopping On-Site", "jj" => "Hopping On-Site"))
```
"""
    mutable struct TightBindingMFT{T, R} 
        ##### TightBinding Model for MFT, and the ParamBlock containing the expected order parameters of MFT
        TightBindingModel   ::  Model
        HoppingBlock        ::  ParamBlock{2, R}
        ##### The ParamBlock tracking information about the Interacting UnitCell, and the corresponding MFT decomposition functions
        InteractionBlock    ::  Vector{ParamBlock{T, Float64}}
        MFTDecomposition    ::  Vector{Function}
        ##### The MFT expectation value of the full interacting Hamiltonian
        MFTEnergy           ::  Vector{Float64}
        ##### The relative scaling b/w different MFT channels
        MFTScaling          ::  Dict{String, Float64}
        ##### The user defined labels of the different MFT channels
        ChannelLabels       ::  Dict{String, String}


        function TightBindingMFT(TightBindingModel::Model, HoppingBlock::ParamBlock{2, R}, InteractionBlock::Vector{ParamBlock{T, Float64}} , MFTDecomposition::Vector{Function} ; ChannelLabels :: Dict{String, String} = Dict{String, String}("ij" => "Hopping", "ii" => "Hopping On-Site", "jj" => "Hopping On-Site")) where {T, R <: Union{Float64, ComplexF64}}

            @warn "`MFTScaling` attribute not passed. Resorting to default values of uniform relative scaling for every channel!"
            MFTScaling      =   Dict{String, Float64}("ij" => 1.0, "ii" => 1.0, "jj" => 1.0) 

            @assert IsSameUnitCell(TightBindingModel.uc, HoppingBlock.uc) "Inconsistency between Tight-Binding Unit Cell and Expectation Unit Cell"

            return new{T, R}(TightBindingModel, HoppingBlock, InteractionBlock, MFTDecomposition, Float64[], MFTScaling, ChannelLabels)
        end

        function TightBindingMFT(TightBindingModel::Model, HoppingBlock::ParamBlock{2, R}, InteractionBlock::Vector{ParamBlock{T, Float64}}, MFTDecomposition::Vector{Function}, MFTScaling::Dict{String, Float64} ; ChannelLabels :: Dict{String, String} = Dict{String, String}("ij" => "Hopping", "ii" => "Hopping On-Site", "jj" => "Hopping On-Site")) where {T, R <: Union{Float64, ComplexF64}}

            @assert IsSameUnitCell(TightBindingModel.uc, HoppingBlock.uc) "Inconsistency between Tight-Binding Unit Cell and Expectation Unit Cell"

            return new{T, R}(TightBindingModel, HoppingBlock, InteractionBlock, MFTDecomposition, Float64[], MFTScaling, ChannelLabels)
        end

        function TightBindingMFT(TightBindingModel::Model, HoppingBlock::ParamBlock{2, R}, InteractionBlock::ParamBlock{T, Float64} , MFTDecomposition::Function ; ChannelLabels :: Dict{String, String} = Dict{String, String}("ij" => "Hopping", "ii" => "Hopping On-Site", "jj" => "Hopping On-Site")) where {T, R <: Union{Float64, ComplexF64}}

            return TightBindingMFT(TightBindingModel, HoppingBlock, ParamBlock{T, Float64}[InteractionBlock], Function[MFTDecomposition] ; ChannelLabels = ChannelLabels)
        end

        function TightBindingMFT(TightBindingModel::Model, HoppingBlock::ParamBlock{2, R}, InteractionBlock::ParamBlock{T, Float64}, MFTDecomposition::Function, MFTScaling::Dict{String, Float64} ; ChannelLabels :: Dict{String, String} = Dict{String, String}("ij" => "Hopping", "ii" => "Hopping On-Site", "jj" => "Hopping On-Site")) where {T, R <: Union{Float64, ComplexF64}}

            return TightBindingMFT(TightBindingModel, HoppingBlock, ParamBlock{T, Float64}[InteractionBlock], Function[MFTDecomposition], MFTScaling; ChannelLabels = ChannelLabels)
        end

    end


@doc """
```julia
GetMFTEnergy(tbMFT::TightBindingMFT{T, R}) --> Float64
```
Returns the total mean-field energy of the model including decomposed interactions.

"""
    function GetMFTEnergy(tbMFT::TightBindingMFT{T, R}) :: Float64 where {T, R}
        ##### /// TODO: Add Free Hopping energies also 
        ##### /// TODO: Test!!!!

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