using MiniChat.Config;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace MiniChat.Model
{

    /// <summary>
    /// Le cœur du Transformer : chaque token "regarde" les tokens précédents
    /// et décide lesquels sont importants pour prédire la suite.
    ///
    /// Pour chaque position on calcule 3 vecteurs :
    ///  - Query (Q) : "qu'est-ce que je cherche ?"
    ///  - Key   (K) : "qu'est-ce que je contiens ?"
    ///  - Value (V) : "qu'est-ce que je transmets si on me choisit ?"
    /// Score = Q · K. Plus le score est haut, plus on prend la Value de ce token.
    ///
    /// "Causal" = un token ne peut pas regarder le futur (sinon il tricherait).
    /// "Multi-têtes" = plusieurs attentions en parallèle, chacune apprend un rôle différent.
    ///
    /// Le calcul (scores, masque, softmax, moyenne des Values) est fait par
    /// functional.scaled_dot_product_attention : même résultat que l'écrire à la main,
    /// mais libtorch choisit un noyau optimisé, beaucoup plus rapide et économe en mémoire sur GPU.
    /// </summary>
    public sealed class CausalSelfAttention : Module<Tensor, Tensor>
    {
        private readonly TorchSharp.Modules.Linear _qkv;
        private readonly TorchSharp.Modules.Linear _projection;
        private readonly TorchSharp.Modules.Dropout _outputDropout;
        private readonly double _attentionDropout;
        private readonly int _heads;
        private readonly int _headDim;

        public CausalSelfAttention(ModelConfig config) : base(nameof(CausalSelfAttention))
        {
            if (config.EmbedDim % config.Heads != 0)
                throw new ArgumentException("EmbedDim doit être divisible par Heads.");

            _heads = config.Heads;
            _headDim = config.EmbedDim / config.Heads;

            // Une seule couche calcule Q, K et V d'un coup (plus rapide que trois couches).
            _qkv = Linear(config.EmbedDim, 3 * config.EmbedDim, hasBias: false);
            _projection = Linear(config.EmbedDim, config.EmbedDim);
            _attentionDropout = config.Dropout;
            _outputDropout = Dropout(config.Dropout);

            RegisterComponents();
        }

        /// <param name="x">Forme (batch, temps, EmbedDim).</param>
        public override Tensor forward(Tensor x)
        {
            long batch = x.shape[0], time = x.shape[1], channels = x.shape[2];

            // 1. Calcul de Q, K, V puis découpage en têtes : (batch, têtes, temps, headDim)
            var parts = _qkv.forward(x).chunk(3, 2);
            var q = SplitHeads(parts[0], batch, time);
            var k = SplitHeads(parts[1], batch, time);
            var v = SplitHeads(parts[2], batch, time);

            // 2 à 5 en un seul appel, équivalent à :
            //   scores  = Q · Kᵀ / sqrt(headDim)            (batch, têtes, temps, temps)
            //   scores  = masque causal (le futur à -∞)     un token ne voit pas la suite
            //   poids   = dropout(softmax(scores))          des pourcentages qui somment à 1
            //   sortie  = poids · V                         moyenne pondérée des Values
            // Arguments par position : dans TorchSharp, le paramètre "causal" s'appelle is_casual (faute de frappe).
            double dropout = training ? _attentionDropout : 0.0;   // pas de dropout en génération
            var attention = functional.scaled_dot_product_attention(q, k, v, null, dropout, true);

            // On recolle les têtes ensemble.
            var output = attention                        // (batch, têtes, temps, headDim)
                .transpose(1, 2).contiguous()             // (batch, temps, têtes, headDim)
                .view(batch, time, channels);             // (batch, temps, EmbedDim)

            return _outputDropout.forward(_projection.forward(output));
        }

        private Tensor SplitHeads(Tensor t, long batch, long time) =>
            t.view(batch, time, _heads, _headDim).transpose(1, 2);
    }
}