using System.Text;
using Parquet;
using Parquet.Schema;

namespace DataBuilder;

/// <summary>
/// Extrait le texte d'une colonne de fichiers .parquet (Wikipedia, FineWeb... sur Hugging Face)
/// pour le pré-entraînement. Chaque valeur de la colonne est un document.
///
/// Le nettoyage est celui de Gutenberg (GutenbergCleaner.CleanParagraph), mais ici une ligne = un paragraphe :
/// ces textes ne sont pas coupés à ~70 caractères comme les livres, et beaucoup séparent leurs paragraphes
/// par un seul saut de ligne. Résultat : un paragraphe par ligne.
///
/// Le texte est écrit dans le fichier au fur et à mesure : un seul fichier Wikipedia peut donner plusieurs Go,
/// trop pour une chaîne .NET (1 milliard de caractères au plus).
///
/// Après ConvertAsync(), les compteurs (Kept, Empty...) disent ce qui a été gardé.
/// </summary>
public sealed class ParquetConverter
{
    private readonly string _column;
    private readonly long? _maxBytes;
    private readonly HashSet<char> _vocabulary = new();

    /// <summary>Documents écrits dans le fichier.</summary>
    public int Kept { get; private set; }

    /// <summary>Caractères écrits dans le fichier.</summary>
    public long Characters { get; private set; }

    /// <summary>Nombre de caractères différents écrits.</summary>
    public int Vocabulary => _vocabulary.Count;

    /// <summary>Documents vides (ou null) après nettoyage.</summary>
    public int Empty { get; private set; }

    /// <summary>Vrai si la lecture s'est arrêtée à la taille maximale, avant la fin des fichiers.</summary>
    public bool ReachedMaxSize { get; private set; }

    /// <param name="column">Nom de la colonne de texte.</param>
    /// <param name="maxBytes">Taille max de la sortie en octets UTF-8 (null = pas de limite).</param>
    public ParquetConverter(string column, long? maxBytes)
    {
        _column = column;
        _maxBytes = maxBytes;
    }

    /// <summary>Écrit le texte nettoyé dans outputPath. Un document qui ferait dépasser la taille max n'est pas gardé.</summary>
    public async Task ConvertAsync(IEnumerable<string> inputPaths, string outputPath)
    {
        // Les colonnes de tous les fichiers d'abord (ça ne lit que l'en-tête) : une erreur sur le dernier
        // fichier arrive tout de suite, pas après avoir lu tous les autres pour rien, et la sortie n'est pas écrasée.
        var columns = new Dictionary<string, string>();
        foreach (string path in inputPaths)
        {
            await using var reader = await ParquetReader.CreateAsync(path);
            columns[path] = FindColumn(reader.Schema, path).Name;
        }

        // UTF-8 sans BOM, fins de ligne "\n" : exactement ce que MiniChat attend.
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(outputPath))!);
        await using var output = new StreamWriter(outputPath, false, new UTF8Encoding(false));
        long bytes = 0;

        foreach (string path in inputPaths)
        {
            await using var reader = await ParquetReader.CreateAsync(path);
            DataField field = reader.Schema.GetDataFields().First(f => f.Name == columns[path]);
            int keptBefore = Kept;

            for (int i = 0; i < reader.RowGroupCount; i++)
            {
                using var rowGroup = reader.OpenRowGroupReader(i);
                var documents = new string?[rowGroup.RowCount];
                await rowGroup.ReadAsync(field, documents.AsMemory());

                foreach (string? document in documents)
                {
                    string cleaned = Clean(document);
                    if (cleaned.Length == 0)
                    {
                        Empty++;
                        continue;
                    }
                    long size = TextCleaner.DocToken.Length + Encoding.UTF8.GetByteCount(cleaned);
                    if (bytes + size > _maxBytes)
                    {
                        ReachedMaxSize = true;
                        Console.WriteLine($"  {Path.GetFileName(path)} : {Kept - keptBefore:N0} documents (taille max atteinte).");
                        return;
                    }

                    // <|doc|> devant chaque document : le modèle voit où commence un nouveau texte.
                    await output.WriteAsync(TextCleaner.DocToken);
                    await output.WriteAsync(cleaned);
                    _vocabulary.UnionWith(cleaned);
                    bytes += size;
                    Characters += cleaned.Length;
                    Kept++;
                }
            }

            Console.WriteLine($"  {Path.GetFileName(path)} : {Kept - keptBefore:N0} documents.");
        }
    }

    /// <summary>Un paragraphe nettoyé par ligne, "" si rien ne reste.</summary>
    private static string Clean(string? document)
    {
        if (string.IsNullOrWhiteSpace(document)) return "";

        var paragraphs = document.Replace("\r\n", "\n").Replace('\r', '\n')
            .Split('\n')
            .Select(GutenbergCleaner.CleanParagraph)
            .Where(p => p.Length > 0);
        string text = string.Join('\n', paragraphs);
        return text.Length > 0 ? text + '\n' : "";
    }

    /// <summary>
    /// La colonne demandée, sinon la seule colonne de texte du fichier s'il n'y en a qu'une
    /// (les jeux de données ne l'appellent pas tous "text" : TinyStories l'appelle "french-tinystories").
    /// </summary>
    private DataField FindColumn(ParquetSchema schema, string path)
    {
        var fields = schema.GetDataFields();
        var requested = fields.FirstOrDefault(f => f.Name == _column && IsText(f));
        if (requested is not null) return requested;

        var textFields = fields.Where(IsText).ToList();
        if (textFields.Count == 1)
        {
            Console.WriteLine($"  {Path.GetFileName(path)} : pas de colonne '{_column}', lecture de la seule colonne de texte '{textFields[0].Name}'.");
            return textFields[0];
        }

        throw new InvalidDataException(
            $"{Path.GetFileName(path)} : pas de colonne de texte '{_column}'. "
            + $"Colonnes : {string.Join(", ", fields.Select(f => $"{f.Name} ({(IsText(f) ? "texte" : f.ClrType.Name)})"))}. "
            + "Choisis-en une avec --column.");
    }

    // Parquet.Net 6 annonce les colonnes de texte en ReadOnlyMemory<char> (string dans les versions d'avant).
    private static bool IsText(DataField field) =>
        field.ClrType == typeof(ReadOnlyMemory<char>) || field.ClrType == typeof(string);
}
