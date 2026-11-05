using System.Runtime.InteropServices;

namespace MiniChat.Training;

/// <summary>
/// Mémoire utilisée sur le GPU.
///
/// TorchSharp ne fournit pas cette information : on la demande directement au runtime CUDA
/// (cudart64_12.dll, livrée avec le paquet TorchSharp-cuda-windows).
/// C'est la mémoire occupée sur toute la carte : MiniChat, mais aussi l'affichage de Windows
/// et les autres programmes. libtorch garde en réserve la mémoire qu'il a utilisée une fois,
/// donc ce chiffre monte jusqu'au pic de l'entraînement puis reste stable.
/// </summary>
public static class GpuMemory
{
    // cudaError_t cudaMemGetInfo(size_t* free, size_t* total) ; 0 = succès.
    [DllImport("cudart64_12")]
    private static extern int cudaMemGetInfo(out nuint free, out nuint total);

    /// <summary>(utilisée, totale) en Mo, ou null si l'information n'est pas disponible.</summary>
    public static (long UsedMb, long TotalMb)? Read()
    {
        try
        {
            if (cudaMemGetInfo(out var free, out var total) != 0)
                return null;
            return ((long)(total - free) / (1024 * 1024), (long)total / (1024 * 1024));
        }
        catch (Exception e) when (e is DllNotFoundException or EntryPointNotFoundException)
        {
            return null;   // pas de runtime CUDA (paquet CPU) : on n'affiche rien
        }
    }
}
