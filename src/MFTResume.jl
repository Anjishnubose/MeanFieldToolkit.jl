module MFTResume
    export ResumeMFT!, ReadMFT

    using FixedPointToolkit, Logging, LinearAlgebra

    using ..MeanFieldToolkit.MFTIter: MFTIterator
    using ..MeanFieldToolkit.MFTRun: SolveMFT!


@doc """
```julia
ResumeMFT!(fileName::String ; Update::Function = SimpleMixing, max_iter::Int64 = 100, tol::Float64 = 1e-6, checkpoint_interval::Int64 = 50) --> SelfCons 
```
Resumes a mean-field simulation from a checkpoint file.
- If `Update` is passed, then the update function is used to perform the self-consistency update.
- If `max_iter` is passed, then the maximum number of iterations is set to `max_iter`.
- If `tol` is passed, then the tolerance for convergence is set to `tol`.
- If `checkpoint_interval` is passed, then the checkpoint interval is set to `checkpoint_interval`.
"""
    function ResumeMFT!(fileName::String ; Update::Function = SimpleMixing, max_iter::Int64 = 100, tol::Float64 = 1e-6, checkpoint_interval::Int64 = 50) :: SelfCons 

        SelfConsParams  =   Dict{Symbol, Real}(:tol => tol, :max_iter => max_iter, :checkpoint_interval => checkpoint_interval)
        
        SC              =   ContinueFixedPoint!(fileName, MFTIterator, Update, SelfConsParams)
        @info "Completed!"

        return SC
    end


@doc """
```julia
ReadMFT(fileName::String) --> Dict{String, Any}
```
Reads a mean-field simulation from a checkpoint file.
Returns a dictionary containing the following keys:
- `Convergence`: The norm of the difference between the input and the output at the last iteration.
- `Expectations`: The expectation values of the order parameters in the last iteration.
- `Iterations`: The number of iterations performed.
- `MFT`: The mean-field theory object used for the simulation.
"""
    function ReadMFT(fileName::String) 

        f   =   read_checkpoint(fileName)

        return Dict("Convergence" => norm(f["inputs"][end] - f["outputs"][end]) / sqrt(length(f["inputs"][end])),
                    "Expectations" => f["outputs"][end],
                    "Iterations" => length(f["inputs"]),
                    "MFT" => f["function args"][1])
    end

end