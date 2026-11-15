using TorchSharp;
using static TorchSharp.torch;

namespace MiniChat.Model;

/// <summary>
/// Plusieurs conversations dans un même extrait (fine-tuning, voir TextDataset) : chacune doit être lue comme si
/// elle était seule. Le numéro de document de chaque token (documents, forme (batch, temps), croissant dans
/// chaque extrait) donne deux choses :
///
///   Attention : un token ne voit que les tokens d'avant DE SA conversation (causal ET même document).
///     documents   0 0 0 1 1          ligne = le token qui regarde, colonne = le token regardé
///                 1 . . . .
///                 1 1 . . .
///                 1 1 1 . .
///                 . . . 1 .          la conversation 1 ne voit rien de la conversation 0
///                 . . . 1 1
///   Positions : elles repartent de 0 à chaque conversation (0 1 2 0 1), pour RoPE (ou l'embedding de position,
///     Classic) : la conversation 1 a les mêmes positions que si elle commençait l'extrait.
///
/// Chaque token se voit toujours lui-même : aucune ligne n'est vide (softmax d'une ligne vide = NaN).
/// </summary>
public sealed record DocumentMask(Tensor Attention, Tensor Positions)
{
    /// <param name="documents">Forme (batch, temps), int64 : le numéro de document de chaque token.</param>
    /// <returns>Attention : forme (batch, 1, temps, temps), booléens (1 = visible), la même pour toutes les têtes.
    /// Positions : forme (batch, temps), int64.</returns>
    public static DocumentMask From(Tensor documents)
    {
        long time = documents.shape[1];
        var device = documents.device;

        var sameDocument = documents.unsqueeze(2).eq(documents.unsqueeze(1));      // (batch, temps, temps)
        var causal = ones(time, time, ScalarType.Bool, device).tril();
        var attention = sameDocument.logical_and(causal).unsqueeze(1);

        // Position = index du token - index du premier token de sa conversation. Les débuts de conversation
        // gardent leur index, les autres 0 : le maximum cumulé donne, à chaque token, le début de la sienne.
        var index = arange(time, ScalarType.Int64, device).unsqueeze(0).expand(documents.shape[0], time);
        var starts = cat(new[]
        {
            ones(documents.shape[0], 1, ScalarType.Bool, device),
            documents.narrow(1, 1, time - 1).ne(documents.narrow(1, 0, time - 1)),
        }, 1);
        var (firstOfDocument, _) = (index * starts.to(ScalarType.Int64)).cummax(1);
        return new DocumentMask(attention, index - firstOfDocument);
    }
}
