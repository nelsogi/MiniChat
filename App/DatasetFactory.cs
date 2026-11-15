using System.Text;
using System.Text.RegularExpressions;
using MiniChat.Chat;
using MiniChat.Data;
using MiniChat.Tokenization;
using MiniChat.UI;
using static TorchSharp.torch;

namespace MiniChat.App;

/// <summary>
/// Les fichiers de data/ et ce qu'on en tire pour l'entraînement : le tokenizer d'un nouveau modèle,
/// et les datasets de chaque phase (texte.txt, dialogues.txt + identite.txt), via le cache de tokens.
/// </summary>
public sealed class DatasetFactory
{
    // Taille du vocabulaire BPE : plus grand = mots entiers plus souvent, mais plus de données nécessaires.
    // 8000 : texte.txt fait près d'un milliard de tokens, de quoi apprendre chaque token ; chaque extrait de
    // BlockSize tokens contient plus de texte. Les ids restent sur 2 octets dans cache/ (jusqu'à 32 767).
    public const int TokenizerVocabSize = 8000;

    // Le tokenizer apprend sur un échantillon de chaque fichier, réparti dans tout le fichier (voir ReadSample),
    // pas sur les fichiers entiers : les fusions apprises ne changent presque plus au-delà. dialogues.txt a
    // sa part à lui, sinon les mots des conversations (questions, "tu", consignes) seraient noyés dans texte.txt.
    // ~1 min 30 pour 20 Mo et 4000 tokens ; compter quelques minutes pour 80 Mo et 8000 tokens.
    private const int TokenizerTextSampleMb = 60;
    private const int TokenizerDialogueSampleMb = 20;

    // identite.txt (qui est MiniChat) est mélangé aux dialogues du fine-tuning, chaque conversation recopiée
    // IdentityRepeats fois à des places tirées au hasard (graine fixe). Une seule copie ferait 0,04 % des
    // dialogues : le modèle ne la verrait que 2 ou 3 fois. 10 copies : ~0,3 %, chaque conversation vue ~25 fois.
    private const int IdentityRepeats = 10;
    private const int IdentitySeed = 7;

    private readonly string _identityPath;
    private readonly string _cacheFolder;        // tokens de texte.txt et dialogues.txt déjà découpés (voir TokenCache)
    private readonly Device _device;

    public DatasetFactory(string dataFolder, string cacheFolder, Device device)
    {
        TextPath = Path.Combine(dataFolder, "texte.txt");
        DialoguesPath = Path.Combine(dataFolder, "dialogues.txt");
        _identityPath = Path.Combine(dataFolder, "identite.txt");
        QuestionsPath = Path.Combine(dataFolder, "questions-test.txt");
        EvalPath = Path.Combine(dataFolder, "eval.json");
        _cacheFolder = cacheFolder;
        _device = device;
    }

    /// <summary>texte.txt : le texte général du pré-entraînement.</summary>
    public string TextPath { get; }

    /// <summary>dialogues.txt (Toi:/IA:) : les conversations du fine-tuning.</summary>
    public string DialoguesPath { get; }

    /// <summary>questions-test.txt : les questions posées après chaque entraînement et par /test.</summary>
    public string QuestionsPath { get; }

    /// <summary>eval.json : les questions notées toutes seules de /eval (voir EvalTest).</summary>
    public string EvalPath { get; }

    /// <summary>
    /// Nouveau tokenizer BPE, appris sur les DEUX fichiers : il connaît les mots des dialogues,
    /// et le modèle garde le même vocabulaire (donc la même taille de sortie) entre les deux phases.
    /// </summary>
    public BpeTokenizer TrainTokenizer() =>
        Messages.Spin($"Entraînement du tokenizer BPE ({TokenizerVocabSize} tokens, sur {TokenizerTextSampleMb} Mo " +
                      $"de texte.txt et {TokenizerDialogueSampleMb} Mo de dialogues.txt au plus)...",
            () => BpeTokenizer.Train(ReadSample(TextPath, TokenizerTextSampleMb) + "\n" + TokenizerDialogues(), TokenizerVocabSize));

    // Les tokens de texte.txt et de dialogues.txt, relus depuis cache/ s'ils y sont pour ce fichier et ce tokenizer
    // (voir TokenCache) : texte.txt n'est découpé qu'une fois, pas à chaque phase ni à chaque lancement.
    public TextDataset GeneralText(BpeTokenizer tokenizer, int seed) =>
        Dataset("texte", [CheckExists(TextPath)], "", tokenizer, seed, onlyReplies: false,
            () => TextDataset.TokenizeFile(TextPath, tokenizer));

    // Les dialogues et identite.txt : le cache change si l'un des deux change (ou IdentityRepeats).
    // Chaque extrait est fait de conversations entières, qui commencent à <|doc|> et ne se voient pas entre elles.
    public TextDataset Dialogues(BpeTokenizer tokenizer, int seed) =>
        Dataset("dialogues", DialogueSources(), $"identite x{IdentityRepeats}, graine {IdentitySeed}", tokenizer, seed,
            onlyReplies: true, () => TextDataset.Tokenize(ReadDialogues(), tokenizer, onlyReplies: true),
            documentToken: tokenizer.SpecialId(BpeTokenizer.DocToken));

    public static string CheckExists(string path) =>
        File.Exists(path) ? path : throw new UserException($"Fichier de données introuvable : {path}");

    private TextDataset Dataset(string name, IReadOnlyList<string> sources, string settings, BpeTokenizer tokenizer,
        int seed, bool onlyReplies, Func<TokenizedText> tokenize, int? documentToken = null)
    {
        var (text, fromCache) = Messages.Spin($"Lecture de {Label(sources)}...",
            () => TokenCache.Load(_cacheFolder, name, sources, settings, tokenizer, onlyReplies, tokenize));
        Messages.Dim($"{Label(sources)} : " + (fromCache ? "tokens relus depuis cache/." : "découpé en tokens, gardé dans cache/."));
        return new TextDataset(text, seed, _device, documentToken: documentToken);
    }

    // Lit un fichier de données. Sur Windows les fins de ligne sont "\r\n" : on garde seulement "\n".
    private static string ReadText(string path) => File.ReadAllText(CheckExists(path)).Replace("\r\n", "\n");

    // "dialogues.txt + identite.txt"
    private static string Label(IReadOnlyList<string> sources) => string.Join(" + ", sources.Select(Path.GetFileName));

    // dialogues.txt, et identite.txt s'il existe.
    private IReadOnlyList<string> DialogueSources()
    {
        if (File.Exists(_identityPath))
            return [CheckExists(DialoguesPath), _identityPath];

        Messages.Warning($"Pas de {Path.GetFileName(_identityPath)} : le modèle n'apprendra pas qui il est (nom, ce qu'il sait faire).");
        return [CheckExists(DialoguesPath)];
    }

    // dialogues.txt (Toi:/IA:) au format du modèle, avec ses tokens spéciaux (voir ChatFormat).
    // Les conversations d'identite.txt y sont glissées IdentityRepeats fois chacune, à des places tirées au hasard
    // (graine fixe : même résultat à chaque fois, donc le cache reste valable).
    private string ReadDialogues()
    {
        var conversations = Conversations(ReadText(DialoguesPath));
        if (File.Exists(_identityPath))
        {
            var identity = Conversations(ReadText(_identityPath));
            var random = new Random(IdentitySeed);
            for (int copy = 0; copy < IdentityRepeats; copy++)
                foreach (string conversation in identity)
                    conversations.Insert(random.Next(conversations.Count + 1), conversation);
        }
        return ChatFormat.FromDialogueFile(string.Join("\n\n", conversations));
    }

    // Les dialogues pour le tokenizer : un échantillon de dialogues.txt (voir ReadSample) et identite.txt en entier,
    // au format du modèle comme à l'entraînement (BpeTokenizer.Train remplace les tokens spéciaux par "\n").
    // L'échantillon commence au milieu d'une conversation : les lignes avant le premier "Toi:" ou "IA:" sont ignorées.
    private string TokenizerDialogues()
    {
        string text = ReadSample(DialoguesPath, TokenizerDialogueSampleMb);
        if (File.Exists(_identityPath))
            text += "\n\n" + ReadText(_identityPath);
        return ChatFormat.FromDialogueFile(text);
    }

    // Les conversations d'un fichier Toi:/IA:, séparées par une ligne vide (comme les lit ChatFormat.FromDialogueFile).
    private static List<string> Conversations(string text) =>
        Regex.Split(text, @"\n[^\S\n]*\n").Select(c => c.Trim('\n')).Where(c => c.Trim().Length > 0).ToList();

    // Un fichier de plus de maxMb Mo : maxMb blocs de 1 Mo pris à intervalles réguliers, coupés aux fins de ligne.
    // Réparti dans tout le fichier, l'échantillon voit chaque source (texte.txt en colle souvent plusieurs bout à bout).
    // Un fichier plus petit est lu en entier.
    private static string ReadSample(string path, int maxMb)
    {
        const int BlockSize = 1 << 20;
        long length = new FileInfo(CheckExists(path)).Length;
        if (length <= (long)maxMb * BlockSize)
            return ReadText(path);

        var sample = new StringBuilder();
        var buffer = new byte[BlockSize];
        using var file = File.OpenRead(path);
        for (int i = 0; i < maxMb; i++)
        {
            file.Position = length / maxMb * i;
            int read = file.ReadAtLeast(buffer, BlockSize, throwOnEndOfStream: false);
            // Le bloc commence et finit au milieu d'une ligne (voire d'un caractère UTF-8) : on garde les lignes entières.
            string block = Encoding.UTF8.GetString(buffer, 0, read);
            int first = block.IndexOf('\n') + 1;
            int last = block.LastIndexOf('\n') + 1;
            if (last > first)
                sample.Append(block, first, last - first);
        }
        return sample.Replace("\r\n", "\n").ToString();
    }
}
