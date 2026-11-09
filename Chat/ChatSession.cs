using MiniChat.Generation;
using MiniChat.Tokenization;

namespace MiniChat.Chat;

/// <summary>
/// Transforme le générateur de texte en chat.
///
/// Astuce : le modèle ne sait pas ce qu'est une "conversation". Il a juste appris des textes où
/// les messages sont balisés par des tokens spéciaux (voir ChatFormat) :
///     &lt;|user|&gt;...&lt;|end|&gt;&lt;|assistant|&gt;...&lt;|end|&gt;
/// Donc on écrit "&lt;|user|&gt;{ton message}&lt;|end|&gt;&lt;|assistant|&gt;" et on le laisse continuer
/// jusqu'à ce qu'il écrive &lt;|end|&gt;. C'est le même principe que les vrais chatbots.
///
/// Le prompt envoyé au modèle :
///     &lt;|doc|&gt;                                     <- une conversation commence
///     &lt;|user|&gt;...&lt;|end|&gt;&lt;|assistant|&gt;...&lt;|end|&gt;    <- les derniers tours COMPLETS qui tiennent en mémoire
///     &lt;|user|&gt;{message}&lt;|end|&gt;                    <- la question
///     &lt;|assistant|&gt;                               <- le modèle continue ici
///
/// Le modèle sait qui il est (MiniChat) par le fine-tuning : data/identite.txt est mélangé aux dialogues.
/// Rien n'est ajouté au prompt : toute la mémoire (BlockSize tokens) reste pour la conversation.
///
/// À l'écran et dans les fichiers (questions de test, /save-chat), les messages gardent les noms Toi et IA.
/// </summary>
public sealed class ChatSession
{
    /// <summary>Noms affichés devant les messages (et préfixes des lignes de dialogues.txt).</summary>
    public const string UserPrefix = "Toi: ";
    public const string BotPrefix = "IA: ";

    // Début de chaque prompt : une conversation commence, comme chaque conversation du fine-tuning.
    private const string Start = BpeTokenizer.DocToken;

    private readonly TextGenerator _generator;
    private readonly HashSet<long> _stopTokens;   // tous les tokens spéciaux : <|end|>, et les autres par sécurité

    // Les tours complets (question, réponse). On ne coupe jamais au milieu.
    private readonly List<(string User, string Reply)> _turns = new();
    private int _turnsInLastPrompt;

    // Réglages de départ. Nel Code les remplace par ceux de settings.json ; les questions de test gardent
    // ces valeurs fixes (sauf la température), pour que leurs résultats restent comparables.
    public const double DefaultTemperature = 0.8;
    public const int DefaultTopK = 20;
    public const double DefaultTopP = 0.9;

    public double Temperature { get; set; } = DefaultTemperature;

    /// <summary>Nombre de tokens candidats à chaque tirage (0 = tous).</summary>
    public int TopK { get; set; } = DefaultTopK;

    /// <summary>Part des chances gardée à chaque tirage (1 = tout).</summary>
    public double TopP { get; set; } = DefaultTopP;

    /// <summary>Longueur max d'une réponse, en tokens (~3 caractères chacun).</summary>
    public int MaxReplyTokens { get; set; } = 80;

    /// <summary>Graine du tirage au sort : même graine = même réponse. null = hasard différent à chaque fois.</summary>
    public long? Seed { get; set; }

    public ChatSession(TextGenerator generator)
    {
        _generator = generator;
        _stopTokens = generator.Tokenizer.SpecialIds.Select(id => (long)id).ToHashSet();
    }

    /// <summary>
    /// Envoie un message et renvoie la réponse, caractère par caractère via onChar.
    /// La réponse s'arrête quand le modèle écrit &lt;|end|&gt; (ou un autre token spécial : &lt;|user|&gt;
    /// voudrait dire qu'il écrit la question suivante à notre place).
    /// cancellation : arrête la réponse au caractère suivant ; le début de réponse reste dans la conversation.
    /// </summary>
    public string Send(string userMessage, Action<char> onChar, CancellationToken cancellation = default)
    {
        string question = ChatFormat.User(userMessage) + BpeTokenizer.AssistantToken;

        string reply = _generator.Generate(BuildPrompt(question), MaxReplyTokens, Temperature, TopK, TopP, c =>
        {
            if (cancellation.IsCancellationRequested) return false;
            onChar(c);
            return true;
        }, Seed, stopTokens: _stopTokens);

        _turns.Add((userMessage.Trim(), reply.Trim()));
        return reply;
    }

    /// <summary>Oublie la conversation (le modèle, lui, ne change pas).</summary>
    public void Reset()
    {
        _turns.Clear();
        _turnsInLastPrompt = 0;
    }

    /// <summary>Le générateur (donc le modèle) qui répond.</summary>
    public TextGenerator Generator => _generator;

    /// <summary>Nombre de tours de la conversation (commande /info).</summary>
    public int TurnCount => _turns.Count;

    /// <summary>Les tours de la conversation, en texte simple (commande /save-chat).</summary>
    public IReadOnlyList<(string User, string Reply)> Turns => _turns;

    /// <summary>Combien de ces tours tenaient dans le dernier prompt (commande /info).</summary>
    public int TurnsInLastPrompt => _turnsInLastPrompt;

    /// <summary>
    /// &lt;|doc|&gt; + les tours les plus récents qui tiennent + la question.
    /// Le modèle ne voit que BlockSize tokens, et il faut laisser MaxReplyTokens pour sa réponse :
    /// on remonte les tours du plus récent au plus ancien, et on s'arrête au premier qui ne tient plus.
    /// Chaque morceau commence et finit par un token spécial, jamais fusionné avec ses voisins :
    /// les tokens des morceaux s'additionnent exactement.
    /// </summary>
    private string BuildPrompt(string question)
    {
        int budget = _generator.Model.Config.BlockSize - MaxReplyTokens
                     - _generator.CountTokens(Start) - _generator.CountTokens(question);

        int first = _turns.Count;
        while (first > 0)
        {
            var (user, reply) = _turns[first - 1];
            int cost = _generator.CountTokens(ChatFormat.Turn(user, reply));
            if (cost > budget) break;
            budget -= cost;
            first--;
        }

        _turnsInLastPrompt = _turns.Count - first;
        return Start + string.Concat(_turns.Skip(first).Select(t => ChatFormat.Turn(t.User, t.Reply))) + question;
    }
}
