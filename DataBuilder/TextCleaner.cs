using System.Text;
using System.Text.RegularExpressions;

namespace DataBuilder;

/// <summary>
/// Nettoyage commun aux deux convertisseurs.
///
/// MiniChat apprend caractère par caractère : chaque caractère différent ajoute une sortie au modèle.
/// On remplace donc les variantes typographiques (’ “ ” … —) par leur version simple,
/// et on garde seulement un alphabet raisonnable (ASCII + lettres accentuées + « » œ).
/// </summary>
public static partial class TextCleaner
{
    /// <summary>
    /// Écrit devant chaque document de texte.txt (un livre, un article) : le tokenizer de MiniChat en fait
    /// un token spécial, et le modèle apprend qu'un nouveau texte commence là. Même texte que
    /// BpeTokenizer.DocToken dans MiniChat (DataBuilder est un projet à part, il ne le voit pas).
    /// </summary>
    public const string DocToken = "<|doc|>";

    private static readonly Dictionary<char, string> Replacements = new()
    {
        ['’'] = "'", ['‘'] = "'", ['ʼ'] = "'",     // apostrophes typographiques
        ['“'] = "\"", ['”'] = "\"", ['„'] = "\"",  // guillemets anglais
        ['…'] = "...",                                       // points de suspension
        ['–'] = "-", ['—'] = "-", ['−'] = "-",     // tirets
        [' '] = " ", [' '] = " ", [' '] = " ",     // espaces insécables / fines
        ['\t'] = " ",
    };

    /// <summary>Caractères que MiniChat a le droit de voir.</summary>
    public static bool IsAllowed(char c) =>
        c is >= ' ' and <= '~'                     // ASCII imprimable
        || c is >= 'À' and <= 'ÿ' and not '×' and not '÷'   // lettres accentuées (Latin-1)
        || c is '«' or '»' or 'œ' or 'Œ' or '\n';

    /// <summary>Remplace les variantes typographiques et recolle les accents décomposés (e + ´ -> é).</summary>
    public static string Normalize(string text)
    {
        var builder = new StringBuilder(text.Length);
        foreach (char c in text.Normalize(NormalizationForm.FormC))
        {
            if (Replacements.TryGetValue(c, out var replacement)) builder.Append(replacement);
            else builder.Append(c);
        }
        return builder.ToString();
    }

    /// <summary>
    /// Répare le « œ » mal encodé de certains jeux de données (son UTF-8 relu en Windows-1252) :
    /// "cÅ“ur" -> "cœur", "Å’ufs" -> "Œufs". Le 2e caractère est parfois devenu " ou ?, un caractère invisible,
    /// ou a disparu ("Åufs") : on répare tout "Å" devant un "u" (œuf, cœur, œuvre, sœur, nœud...).
    /// Seulement pour les dialogues : dans les textes, "Å" existe vraiment ("Ålesund").
    /// </summary>
    public static string FixBrokenOe(string text) =>
        !text.Contains('Å') ? text
        : BrokenOe().Replace(BrokenCapitalOe().Replace(text, "Œ"), "œ");

    [GeneratedRegex("Å[’'\u0092]")]
    private static partial Regex BrokenCapitalOe();

    [GeneratedRegex("Å[“\"?«\u0080-\u009F�]?(?=u|il)|Å[“\u0093]")]
    private static partial Regex BrokenOe();

    /// <summary>
    /// Retire la mise en forme Markdown des réponses générées par d'autres IA : gras (**mot**, __mot__),
    /// titres (## Titre), séparateurs (---, ***, ####), puces en début de ligne (- , * ).
    /// "## Conseils\n- **Dormir** tôt" -> "Conseils\nDormir tôt". À appeler avant SingleLine (il lit les débuts de ligne).
    /// </summary>
    public static string StripMarkdown(string text) =>
        Bold().Replace(Rules().Replace(LineMarks().Replace(text, ""), " "), "");

    /// <summary>Formule LaTeX (\(...\), \[...\], $$...$$, \frac...) : illisible une fois sur une ligne pour un si petit modèle.</summary>
    public static bool HasLatex(string text) => Latex().IsMatch(text);

    [GeneratedRegex(@"^[ \t]*(#{1,6}[ \t]+|[-*+][ \t]+)", RegexOptions.Multiline)]
    private static partial Regex LineMarks();

    [GeneratedRegex(@"^[ \t]*([-*_#=])\1{2,}[ \t]*$|[ \t](-{3,}|\*{3,}|#{3,})(?=[ \t]|$)", RegexOptions.Multiline)]
    private static partial Regex Rules();

    [GeneratedRegex(@"\*\*|__")]
    private static partial Regex Bold();

    [GeneratedRegex(@"\\\(|\\\[|\$\$|\\(frac|sqrt|binom|times|cdot|sum|int|left|right)\b")]
    private static partial Regex Latex();

    /// <summary>Met tout sur une ligne et réduit les espaces multiples à un seul.</summary>
    public static string SingleLine(string text) => Spaces().Replace(text, " ").Trim();

    public static string RemoveForbidden(string text) => new(text.Where(IsAllowed).ToArray());

    /// <summary>
    /// Coupe le texte après la dernière fin de phrase (. ! ?) qui tient dans maxLength caractères.
    /// Renvoie null s'il n'y en a aucune : couper au milieu d'une phrase apprendrait au modèle à s'arrêter n'importe où.
    /// Exemple (max 20) : "Bonjour. Je suis une IA très bavarde." -> "Bonjour."
    /// </summary>
    public static string? CutAtSentenceEnd(string text, int maxLength)
    {
        if (text.Length <= maxLength) return text;

        for (int i = maxLength - 1; i > 0; i--)
        {
            if (IsSentenceEnd(text, i))
                return text[..(i + 1)];
        }
        return null;
    }

    /// <summary>
    /// Garde au plus maxSentences phrases.
    /// Exemple (2) : "Oui. C'est vrai ! Et en plus..." -> "Oui. C'est vrai !"
    /// </summary>
    public static string FirstSentences(string text, int maxSentences, out bool cut)
    {
        int found = 0;
        for (int i = 0; i < text.Length; i++)
        {
            if (IsSentenceEnd(text, i) && ++found == maxSentences)
            {
                cut = i + 1 < text.Length;
                return text[..(i + 1)];
            }
        }
        cut = false;
        return text;
    }

    /// <summary>
    /// Une fin de phrase = . ! ou ? suivi d'un espace (ou de la fin du texte).
    /// "..." ne compte qu'une fois : seul le dernier point est suivi d'un espace.
    /// </summary>
    // ponytail: "M. Dupont" compte comme une fin de phrase, assez rare pour l'ignorer.
    private static bool IsSentenceEnd(string text, int i) =>
        text[i] is '.' or '!' or '?' && (i + 1 == text.Length || text[i + 1] == ' ') && !IsListNumber(text, i);

    /// <summary>
    /// Le point de "2. " dans une liste numérotée ("1. Pommes 2. Poires") : un nombre de 1 ou 2 chiffres
    /// seul devant le point. Ce n'est pas une fin de phrase (sinon la réponse finirait par "2.").
    /// "en 1950." reste une fin de phrase : 4 chiffres.
    /// </summary>
    private static bool IsListNumber(string text, int i)
    {
        if (text[i] != '.') return false;
        int start = i;
        while (start > 0 && char.IsAsciiDigit(text[start - 1])) start--;
        int digits = i - start;
        return digits is 1 or 2 && (start == 0 || text[start - 1] == ' ');
    }

    /// <summary>
    /// Lit un fichier texte. Les livres anciens sont parfois en Latin-1 au lieu d'UTF-8 :
    /// si la lecture UTF-8 donne des caractères invalides (�), on relit en Latin-1.
    /// </summary>
    public static string ReadFile(string path)
    {
        string text = File.ReadAllText(path, Encoding.UTF8);
        if (text.Contains('�'))
            text = File.ReadAllText(path, Encoding.Latin1);
        return text.Replace("\r\n", "\n").Replace('\r', '\n');
    }

    [GeneratedRegex(@"\s+")]
    private static partial Regex Spaces();
}
