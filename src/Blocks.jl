module Blocks
    export ParamBlock, UpdateBlock!, GetAllBondParams

    using TightBindingToolkit

    mutable struct ParamBlock{T, R}

        params      ::  Vector{Param{T, R}}
        uc          ::  UnitCell{T}
        lookup      ::  Dict{Tuple, Array{ComplexF64, T}}

        ##### ///TODO Make this CreateUnitCell instead of ModifyUnitCell
        function ParamBlock(params::Vector{Param{T, R}}, uc::UnitCell{T}) where {T, R}
            CreateUnitCell!(uc, params)
            lookup  =   Lookup(uc)
            return new{T, R}(params, uc, lookup)
        end
    end


    function UpdateBlock!(Block::ParamBlock{T, R}) where {T, R}
        ModifyUnitCell!(Block.uc, Block.params)
        Block.lookup    =   Lookup(Block.uc)
    end


    function GetAllBondParams(uc::UnitCell{T}, dist::Float64, label::String, mat::Array{ComplexF64, T} ; phases::Vector{ComplexF64} = [1.0, im], checkOffsetRange::Int64 =2) :: Vector{Param{T}} where {T}

        MotherParam     =   Param(1.0, T)
        AddIsotropicBonds!(MotherParam, uc, dist, mat, label ; checkOffsetRange = checkOffsetRange)

        params  =   Param{T, Float64}[]

        for bond in MotherParam.unitBonds
            bondKey     =   (bond.base, bond.target, bond.offset)
            for phase in phases
                param   =   Param(1.0, T)
                AddAnisotropicBond!(param, uc, bond.base, bond.target, bond.offset, phase .* bond.mat, bond.dist, "$(bond.label) : $(round(phase, digits = 2)), $(bondKey)")

                push!(params, param)
            end
        end

        return params
    end











end