module MFTRun
    export SolveMFT!

    using FixedPointToolkit, Distributions, TightBindingToolkit

    using ..MeanFieldToolkit.TBMFT: TightBindingMFT
    using ..MeanFieldToolkit.BDGMFT: BdGMFT
    using ..MeanFieldToolkit.MFTIter: MFTIterator


    function SolveMFT!(mft::TightBindingMFT ; OnSiteMatrices::Vector{Matrix{ComplexF64}} = SpinMats((mft.TightBindingModel.uc.localDim-1)//2), Update::Function = SimpleMixing, Update_kwargs::Dict{Symbol, Any} = Dict{Symbol, Any}(:alpha => 0.5), max_iter::Int64 = 100, tol::Float64 = 1e-6, Initial_range::Tuple{Float64, Float64} = (-0.5, 0.5)) :: SelfCons

        Initial     =   rand(Uniform(Initial_range...), length(mft.HoppingBlock.params))
        selfcons    =   SelfCons(MFTIterator, Update, Initial ; F_args = (mft , ), F_kwargs = Dict(:OnSiteMatrices => OnSiteMatrices), Update_kwargs = Update_kwargs)
        
        FixedPoint!(selfcons ; max_iter = max_iter, tol = tol)
        return selfcons
    end

    function SolveMFT!(mft::BdGMFT ; OnSiteMatrices::Vector{Matrix{ComplexF64}} = SpinMats((mft.TightBindingModel.uc.localDim-1)//2), Update::Function = SimpleMixing, Update_kwargs::Dict{Symbol, Any} = Dict{Symbol, Any}(:alpha => 0.5), max_iter::Int64 = 100, tol::Float64 = 1e-6, Initial_range::Tuple{Float64, Float64} = (-0.5, 0.5)) :: SelfCons

        Initial     =   rand(Uniform(Initial_range...), length(mft.HoppingBlock.params) + length(mft.PairingBlock.params))
        selfcons    =   SelfCons(MFTIterator, Update, Initial ; F_args = (mft , ), F_kwargs = Dict(:OnSiteMatrices => OnSiteMatrices), Update_kwargs = Update_kwargs)
        
        FixedPoint!(selfcons ; max_iter = max_iter, tol = tol)
        return selfcons
    end


    function SolveMFT!(mft::TightBindingMFT, fileName::String ; OnSiteMatrices::Vector{Matrix{ComplexF64}} = SpinMats((mft.TightBindingModel.uc.localDim-1)//2), Update::Function = SimpleMixing, Update_kwargs::Dict{Symbol, Any} = Dict{Symbol, Any}(:alpha => 0.5), max_iter::Int64 = 100, tol::Float64 = 1e-6, checkpoint_interval::Int64 = 50, Initial_range::Tuple{Float64, Float64} = (-0.5, 0.5)) :: SelfCons

        Initial     =   rand(Uniform(Initial_range...), length(mft.HoppingBlock.params))
        selfcons    =   SelfCons(MFTIterator, Update, Initial ; F_args = (mft , ), F_kwargs = Dict(:OnSiteMatrices => OnSiteMatrices), Update_kwargs = Update_kwargs)
        
        FixedPoint!(selfcons, fileName ; max_iter = max_iter, tol = tol, checkpoint_interval = checkpoint_interval)
        return selfcons
    end

    function SolveMFT!(mft::BdGMFT, fileName::String ; OnSiteMatrices::Vector{Matrix{ComplexF64}} = SpinMats((mft.bdgModel.uc_hop.localDim-1)//2), Update::Function = SimpleMixing, Update_kwargs::Dict{Symbol, Any} = Dict{Symbol, Any}(:alpha => 0.5), max_iter::Int64 = 100, tol::Float64 = 1e-6, checkpoint_interval::Int64 = 50, Initial_range::Tuple{Float64, Float64} = (-0.5, 0.5)) :: SelfCons

        Initial     =   rand(Uniform(Initial_range...), length(mft.HoppingBlock.params) + length(mft.PairingBlock.params))
        selfcons    =   SelfCons(MFTIterator, Update, Initial ; F_args = (mft , ), F_kwargs = Dict(:OnSiteMatrices => OnSiteMatrices), Update_kwargs = Update_kwargs)
        
        FixedPoint!(selfcons, fileName ; max_iter = max_iter, tol = tol, checkpoint_interval = checkpoint_interval)
        return selfcons
    end


    function SolveMFT!(mft::T, Initial::Vector{Float64}; OnSiteMatrices::Vector{Matrix{ComplexF64}} = SpinMats((mft.TightBindingModel.uc.localDim-1)//2), Update::Function = SimpleMixing, Update_kwargs::Dict{Symbol, Any} = Dict{Symbol, Any}(:alpha => 0.5), max_iter::Int64 = 100, tol::Float64 = 1e-6) :: SelfCons where {T<:Union{TightBindingMFT, BdGMFT}}

        selfcons    =   SelfCons(MFTIterator, Update, Initial ; F_args = (mft , ), F_kwargs = Dict(:OnSiteMatrices => OnSiteMatrices), Update_kwargs = Update_kwargs)
        
        FixedPoint!(selfcons ; max_iter = max_iter, tol = tol)
        return selfcons
    end

    function SolveMFT!(mft::T, Initial::Vector{Float64}, fileName::String; OnSiteMatrices::Vector{Matrix{ComplexF64}} = SpinMats((mft.TightBindingModel.uc.localDim-1)//2), Update::Function = SimpleMixing, Update_kwargs::Dict{Symbol, Any} = Dict{Symbol, Any}(:alpha => 0.5), max_iter::Int64 = 100, tol::Float64 = 1e-6, checkpoint_interval::Int64 = 50) :: SelfCons where {T<:Union{TightBindingMFT, BdGMFT}}

        selfcons    =   SelfCons(MFTIterator, Update, Initial ; F_args = (mft , ), F_kwargs = Dict(:OnSiteMatrices => OnSiteMatrices), Update_kwargs = Update_kwargs)
        
        FixedPoint!(selfcons, fileName ; max_iter = max_iter, tol = tol, checkpoint_interval = checkpoint_interval)
        return selfcons
    end



end