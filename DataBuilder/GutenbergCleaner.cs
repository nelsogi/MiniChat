using System.Text.RegularExpressions;

namespace DataBuilder;

/// <summary>
/// Nettoie un livre du Projet Gutenberg (fichier .txt) pour le pré-entraînement :
///   1. enlève l'en-tête et la licence (tout ce qui est avant "*** START OF" et après "*** END OF"),
///   2. recolle les paragraphes (Gutenberg coupe les lignes à ~70 caractères),
///   3. normalise la typographie et retire les caractères exotiques.
/// Résultat : un paragraphe par ligne.
/// </summary>
public static partial class GutenbergCleaner
{
    public static string Clean(string book, string name)
    {
        string body = ExtractBody(book, name);
        var paragraphs = new List<string>();

        foreach (string block in BlankLines().Split(body))
        {
            string paragraph = TextCleaner.Normalize(block);
            paragraph = paragraph.Replace("_", "");   // Gutenberg écrit l'italique _comme ceci_
            paragraph = TextCleaner.SingleLine(TextCleaner.RemoveForbidden(paragraph.Replace('\n', ' ')));

            if (paragraph.Length > 0)
                paragraphs.Add(paragraph);
        }

        return string.Join('\n', paragraphs) + '\n';
    }

    /// <summary>Garde le texte entre les marqueurs START et END, ou tout le fichier s'ils sont absents.</summary>
    private static string ExtractBody(string book, string name)
    {
        var start = StartMarker().Match(book);
        var end = EndMarker().Match(book);

        if (!start.Success || !end.Success || end.Index < start.Index)
        {
            Console.WriteLine($"  Attention : marqueurs Gutenberg introuvables dans {name}, tout le fichier est gardé.");
            return book;
        }

        int bodyStart = start.Index + start.Length;
        return book[bodyStart..end.Index];
    }

    // La ligne entière du marqueur, ex. "*** START OF THE PROJECT GUTENBERG EBOOK LES MISÉRABLES ***".
    [GeneratedRegex(@"^\*{3}\s*START OF.*$", RegexOptions.Multiline | RegexOptions.IgnoreCase)]
    private static partial Regex StartMarker();

    [GeneratedRegex(@"^\*{3}\s*END OF", RegexOptions.Multiline | RegexOptions.IgnoreCase)]
    private static partial Regex EndMarker();

    [GeneratedRegex(@"\n[ \t]*\n")]
    private static partial Regex BlankLines();
}
