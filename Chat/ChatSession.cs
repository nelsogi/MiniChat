using System.Text;
using MiniChat.Generation;
using MiniChat.Training;
using TorchSharp;

namespace MiniChat.Chat;

/// <summary>
/// Transforme le générateur de texte en chat.
///
/// Astuce : le modèle ne sait pas ce qu'est une "conversation". Il a juste appris
/// des textes au format :
///     Toi: ...
///     IA: ...
/// Donc on écrit "Toi: {ton message}\nIA: " et on le laisse continuer
/// jusqu'au retour à la ligne. C'est le même principe que les vrais chatbots.
///
/// Le prompt envoyé au modèle :
///     IA: Je suis MiniChat, ...        <- l'identité, toujours en premier
///     Toi: ... / IA: ...               <- les derniers tours COMPLETS qui tiennent en mémoire
///     Toi: {message}                   <- la question
///     IA:                              <- le modèle continue ici
/// </summary>
public sealed class ChatSession
{
    public const string UserPrefix = "Toi: ";
    public const string BotPrefix = "IA: ";

    /// <summary>Phrase d'identité placée au début de chaque conversation : le modèle "sait" qui il est.</summary>
    public const string Identity = BotPrefix + "Je suis MiniChat, un assistant qui répond en français.\n";

    private readonly TextGenerator _generator;

    // Un élément = un tour complet "Toi: ...\nIA: ...\n". On ne coupe jamais au milieu.
    private readonly List<string> _turns = new();
    private int _turnsInLastPrompt;

    public double Temperature { get; set; } = 0.8;

    /// <summary>Nombre de tokens candidats à chaque tirage (0 = tous).</summary>
    public int TopK { get; set; } = 20;

    /// <summary>Part des chances gardée à chaque tirage (1 = tout).</summary>
    public double TopP { get; set; } = 0.9;

    /// <summary>Longueur max d'une réponse, en tokens (~3 caractères chacun).</summary>
    public int MaxReplyTokens { get; set; } = 80;

    /// <summary>Graine du tirage au sort : même graine = même réponse. null = hasard différent à chaque fois.</summary>
    public long? Seed { get; set; }

    public ChatSession(TextGenerator generator) => _generator = generator;

    /// <summary>Envoie un message et renvoie la réponse, caractère par caractère via onChar.</summary>
    public string Send(string userMessage, Action<char> onChar)
    {
        string question = $"{UserPrefix}{userMessage.Trim()}\n{BotPrefix}";

        string reply = _generator.Generate(BuildPrompt(question), MaxReplyTokens, Temperature, TopK, TopP, c =>
        {
            if (c == '\n') return false;   // fin de la réponse de l'IA
            onChar(c);
            return true;
        }, Seed);

        _turns.Add($"{question}{reply}\n");
        return reply;
    }

    /// <summary>Oublie la conversation (le modèle, lui, ne change pas). L'identité reste.</summary>
    public void Reset()
    {
        _turns.Clear();
        _turnsInLastPrompt = 0;
    }

    /// <summary>Taille du modèle, device et réglages de génération (commande /info).</summary>
    public string Info()
    {
        var model = _generator.Model;
        var config = model.Config;
        bool gpu = _generator.Device.type == DeviceType.CUDA;

        var info = new StringBuilder();
        info.AppendLine($"Modèle      : {model.ParameterCount:N0} paramètres | {config.Layers} couches, {config.Heads} têtes, " +
                        $"EmbedDim {config.EmbedDim} | mémoire {config.BlockSize} tokens | vocabulaire {config.VocabSize} tokens");

        string memory = gpu && GpuMemory.Read() is var (used, total) ? $" | mémoire GPU {used:N0} / {total:N0} Mo" : "";
        info.AppendLine($"Calcul      : {(gpu ? "GPU NVIDIA (CUDA)" : "CPU")}{memory}");

        info.AppendLine($"Génération  : température {Temperature} | top-k {(TopK == 0 ? "désactivé" : TopK)} | " +
                        $"top-p {(TopP >= 1 ? "désactivé" : TopP)} | réponse max {MaxReplyTokens} tokens | " +
                        $"graine {(Seed is null ? "aléatoire" : Seed)}");
        info.Append($"Conversation : {_turns.Count} tour(s), dont {_turnsInLastPrompt} dans le dernier prompt (+ l'identité)");
        return info.ToString();
    }

    /// <summary>
    /// Identité + les tours les plus récents qui tiennent + la question.
    /// Le modèle ne voit que BlockSize tokens, et il faut laisser MaxReplyTokens pour sa réponse :
    /// on remonte les tours du plus récent au plus ancien, et on s'arrête au premier qui ne tient plus.
    /// Le BPE coupe toujours sur "\n" et chaque morceau finit par "\n" : les tokens s'additionnent exactement.
    /// </summary>
    private string BuildPrompt(string question)
    {
        int budget = _generator.Model.Config.BlockSize - MaxReplyTokens
                     - _generator.CountTokens(Identity) - _generator.CountTokens(question);

        int first = _turns.Count;
        while (first > 0)
        {
            int cost = _generator.CountTokens(_turns[first - 1]);
            if (cost > budget) break;
            budget -= cost;
            first--;
        }

        _turnsInLastPrompt = _turns.Count - first;
        return Identity + string.Concat(_turns.Skip(first)) + question;
    }
}
