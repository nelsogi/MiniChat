using System.Text;

namespace DataBuilder;

/// <summary>
/// Base des convertisseurs de dialogues (OpenAssistant, JSONL de conversations) : nettoyage des messages,
/// format de sortie de MiniChat et compteurs du bilan.
///     Toi: question
///     IA: réponse
///                          <- une ligne vide entre deux conversations
///     Toi: question
///
/// Après Convert(), les compteurs (Kept, Rejected...) disent ce qui a été gardé ou rejeté, et pourquoi.
/// </summary>
public abstract class DialogueConverter
{
    // Raisons de rejet, affichées telles quelles dans le bilan.
    protected const string NoReply = "pas de réponse utilisable";
    protected const string HasCode = "contient du code";
    protected const string HasLatex = "contient une formule LaTeX";
    protected const string ExoticChars = "caractères exotiques (emoji...)";
    protected const string QuestionTooLong = "question trop longue";
    protected const string NoSentenceEnd = "réponse trop longue, sans fin de phrase où couper";
    protected const string Empty = "message vide";
    private const string Duplicate = "doublon d'une conversation déjà gardée";

    private readonly StringBuilder _output = new();
    private readonly HashSet<string> _seen = new();

    /// <summary>Longueur max d'un message : MiniChat ne voit que 128 caractères à la fois.</summary>
    public int MaxLength { get; }

    /// <summary>Nombre max de phrases gardées dans une réponse de l'IA.</summary>
    public int MaxSentences { get; }

    /// <summary>Réponses de l'IA raccourcies à MaxSentences phrases.</summary>
    public int LimitedReplies { get; private set; }

    /// <summary>Conversations écrites dans le fichier.</summary>
    public int Kept { get; private set; }

    /// <summary>Conversations gardées mais arrêtées avant la fin (un message plus loin était inutilisable).</summary>
    public int Shortened { get; private set; }

    /// <summary>Réponses de l'IA trop longues, coupées à la fin d'une phrase.</summary>
    public int TruncatedReplies { get; private set; }

    /// <summary>Conversations rejetées, par raison.</summary>
    public Dictionary<string, int> Rejected { get; } = new();

    /// <summary>Messages ou lignes écartés dès la lecture, par raison.</summary>
    public Dictionary<string, int> IgnoredMessages { get; } = new();

    protected DialogueConverter(int maxLength, int maxSentences)
    {
        MaxLength = maxLength;
        MaxSentences = maxSentences;
    }

    /// <returns>Le texte au format Toi:/IA:.</returns>
    public abstract string Convert(IEnumerable<string> inputPaths);

    /// <summary>Le texte écrit jusqu'ici par Add().</summary>
    protected string Output => _output.ToString();

    /// <summary>
    /// Écrit une conversation (lignes "Toi: ..." / "IA: ..." qui alternent, en commençant par Toi),
    /// ou la compte comme rejetée.
    /// StopReason = pourquoi elle s'arrête avant la fin (null si on est allé au bout).
    /// </summary>
    protected void Add(List<string> lines, string? stopReason)
    {
        // Une conversation doit finir par une réponse de l'IA.
        if (lines.Count > 0 && lines[^1].StartsWith("Toi: "))
            lines.RemoveAt(lines.Count - 1);

        if (lines.Count < 2)   // il faut au moins une question et une réponse
        {
            Count(Rejected, stopReason ?? NoReply);
            return;
        }

        // Beaucoup de jeux de données (et leurs traductions) contiennent les mêmes conversations plusieurs fois.
        if (!_seen.Add(string.Join('\n', lines)))
        {
            Count(Rejected, Duplicate);
            return;
        }

        // Une ligne vide entre deux conversations : MiniChat y met un <|doc|> (une nouvelle conversation commence).
        if (_output.Length > 0) _output.Append('\n');
        foreach (var line in lines) _output.Append(line).Append('\n');
        Kept++;
        if (stopReason is not null) Shortened++;
    }

    /// <summary>Texte prêt pour MiniChat, ou null avec la raison du rejet.</summary>
    protected (string? Text, string? Rejection) Clean(string rawText, bool fromUser)
    {
        // Le code informatique n'a aucun sens pour un si petit modèle.
        if (rawText.Contains("```")) return (null, HasCode);
        if (TextCleaner.HasLatex(rawText)) return (null, HasLatex);

        // Une réplique = une ligne : MiniChat s'arrête au premier retour à la ligne.
        // Le Markdown (gras, titres, puces) des réponses d'autres IA est retiré avant : il ne s'affiche pas dans Nel Code.
        string text = TextCleaner.SingleLine(TextCleaner.StripMarkdown(TextCleaner.Normalize(TextCleaner.FixBrokenOe(rawText))));

        if (text.Length == 0) return (null, Empty);

        // Caractère exotique (emoji, chinois...) : on préfère ignorer le message que le déformer.
        if (!text.All(TextCleaner.IsAllowed)) return (null, ExoticChars);

        // Réponses courtes : un si petit modèle perd le fil sur une longue réponse.
        // (Les questions ne sont jamais coupées : ça changerait leur sens.)
        if (!fromUser)
        {
            text = TextCleaner.FirstSentences(text, MaxSentences, out bool limited);
            if (limited) LimitedReplies++;
        }

        if (text.Length <= MaxLength) return (text, null);

        // Trop long. Une question coupée changerait de sens : on la rejette.
        // Une réponse, elle, reste valable si on s'arrête à la fin d'une phrase.
        if (fromUser) return (null, QuestionTooLong);

        string? cut = TextCleaner.CutAtSentenceEnd(text, MaxLength);
        if (cut is null) return (null, NoSentenceEnd);

        TruncatedReplies++;
        return (cut, null);
    }

    protected static void Count(Dictionary<string, int> counters, string reason) =>
        counters[reason] = counters.GetValueOrDefault(reason) + 1;
}
