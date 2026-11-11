using System.Text.Json;
using System.Text.Json.Nodes;
using MiniChat.Chat;
using MiniChat.Model;
using MiniChat.Tokenization;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace MiniChat.Storage;

/// <summary>
/// reference.json du .nelmodel (voir NELMODEL-FORMAT.md) : ce que MiniChat calcule avec ce modèle, pour qu'une
/// autre implémentation (Python, Rust...) vérifie qu'elle donne la même chose :
///   - tokenizer : les ids de 5 textes fixes (accents, chiffres, retours à la ligne, espaces, tokens spéciaux...),
///   - modèle : pour 3 prompts de chat, les 10 tokens les plus probables après le prompt, et 30 tokens écrits
///     en mode glouton (toujours le plus probable).
/// Calculé sur CPU en float32, sans cache KV : toute la séquence repasse dans le modèle à chaque token.
/// </summary>
public static class NelModelReference
{
    public const string FileName = "reference.json";

    /// <summary>Les textes du tokenizer. Le dernier a un emoji : 2 unités UTF-16, absentes de l'alphabet.</summary>
    public static readonly string[] Texts =
    {
        "Bonjour, comment vas-tu ?",
        "L'été 2024, les élèves ont lu 3 livres… « Très bien ! »",
        "Première ligne.\nDeuxième ligne,   avec trois espaces.\n\nFin\t!",
        "<|doc|><|user|>Qui es-tu ?<|end|><|assistant|>Je suis MiniChat.<|end|>",
        "Œuf, naïve, cœur — 12,5 % ; x² = 4. 🙂",
    };

    /// <summary>Les messages des prompts de chat (premier tour d'une conversation, voir Prompt).</summary>
    public static readonly string[] ChatMessages = { "Bonjour !", "Qui es-tu ?", "Quelle est la capitale de la France ?" };

    public const int TopTokens = 10;
    public const int GreedyTokens = 30;

    /// <summary>Le prompt du premier message d'une conversation, comme le construit ChatSession.</summary>
    public static string Prompt(string message) => BpeTokenizer.DocToken + ChatFormat.User(message) + BpeTokenizer.AssistantToken;

    private static readonly JsonSerializerOptions Options = new()
    {
        WriteIndented = true,
        Encoder = System.Text.Encodings.Web.JavaScriptEncoder.UnsafeRelaxedJsonEscaping,
    };

    /// <summary>Calcule reference.json pour ce modèle (sur le CPU) et l'écrit dans path.</summary>
    public static void Write(MiniGpt model, BpeTokenizer tokenizer, string path)
    {
        var texts = new JsonArray(Texts.Select(text =>
        {
            long[] ids = tokenizer.Encode(text);
            return (JsonNode)new JsonObject
            {
                ["text"] = text,
                ["ids"] = Ids(ids),
                ["tokens"] = new JsonArray(ids.Select(id => (JsonNode?)tokenizer.Decode(id)).ToArray()),
            };
        }).ToArray());

        var prompts = new JsonArray(ChatMessages.Select(message => (JsonNode)Chat(model, tokenizer, message)).ToArray());

        var reference = new JsonObject
        {
            ["note"] = "Calculé par MiniChat sur CPU, float32, sans cache KV. Probabilités : softmax des logits (température 1). " +
                       "Glouton : argmax à chaque pas, exactement 30 tokens, sans s'arrêter aux tokens spéciaux.",
            ["texts"] = texts,
            ["chat"] = prompts,
        };
        File.WriteAllText(path, reference.ToJsonString(Options));
    }

    private static JsonObject Chat(MiniGpt model, BpeTokenizer tokenizer, string message)
    {
        string prompt = Prompt(message);
        long[] promptIds = tokenizer.Encode(prompt);

        model.eval();
        using var noGrad = no_grad();
        using var scope = NewDisposeScope();

        // Les 10 premiers tokens possibles après le prompt.
        var probabilities = functional.softmax(LastLogits(model, promptIds), -1);
        var (top, topIds) = probabilities.topk(TopTokens);
        var topTokens = new JsonArray(Enumerable.Range(0, TopTokens).Select(i =>
        {
            long id = topIds[i].item<long>();
            return (JsonNode)new JsonObject
            {
                ["id"] = id,
                ["token"] = tokenizer.Decode(id),
                ["probability"] = Math.Round(top[i].item<float>(), 6),
            };
        }).ToArray());

        // Glouton : le plus probable à chaque pas (le premier est donc topTokens[0]).
        var ids = new List<long>(promptIds);
        var generated = new List<long>();
        for (int i = 0; i < GreedyTokens && ids.Count < model.Config.BlockSize; i++)
        {
            using var step = NewDisposeScope();
            long next = LastLogits(model, ids.ToArray()).argmax().item<long>();
            generated.Add(next);
            ids.Add(next);
        }

        return new JsonObject
        {
            ["message"] = message,
            ["prompt"] = prompt,
            ["promptIds"] = Ids(promptIds),
            ["topTokens"] = topTokens,
            ["greedyIds"] = Ids(generated),
            ["greedyText"] = tokenizer.Decode(generated),
        };
    }

    // Les logits après la dernière position de ids, forme (VocabSize).
    private static Tensor LastLogits(MiniGpt model, long[] ids) =>
        model.forward(tensor(ids, device: CPU).unsqueeze(0))[0, -1];

    private static JsonArray Ids(IEnumerable<long> ids) => new(ids.Select(id => (JsonNode?)id).ToArray());
}
