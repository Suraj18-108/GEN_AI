// api/generate.js - Vercel Serverless Function for GPT-2 Story Generation

const STYLE_PROMPTS = {
  "Fantasy": "Write an epic fantasy story filled with magic, mythical creatures, and ancient mystery.",
  "Mystery": "Write a suspenseful noir mystery story steeped in secrets, investigation, and tension.",
  "Science Fiction": "Write a visionary science fiction story exploring future technology and space wonders.",
  "Horror": "Write a chilling psychological horror story where dread and eerie darkness lurk.",
  "Adventure": "Write an exciting adventure story full of danger and courageous exploration.",
  "Romance": "Write an emotionally evocative romance story highlighting deep affection and connection.",
  "Comedy": "Write a lighthearted and witty comedy story filled with humorous misunderstandings.",
  "Thriller": "Write a fast-paced psychological thriller filled with urgency, twists, and danger.",
  "Fairy Tale": "Write an enchanting fairy tale with magical charm, kingdoms, and timeless wonder.",
  "Historical Fiction": "Write an authentic historical fiction story reflecting the drama of a past era."
};

function estimateTokenCount(text) {
  if (!text) return 0;
  // BPE average token estimation ~ 1 token per 4 characters or ~1.3 tokens per word
  const words = text.trim().split(/\s+/).filter(Boolean);
  return Math.max(1, Math.round(words.length * 1.33));
}

function calculateLexicalDiversity(text) {
  if (!text) return 0;
  const words = text.toLowerCase().match(/\b\w+\b/g) || [];
  if (words.length === 0) return 0;
  const uniqueWords = new Set(words);
  return Number((uniqueWords.size / words.length).toFixed(3));
}

function formatStory(opening, continuation) {
  const cleanOpening = opening.trim();
  let cleanCont = (continuation || "").trim();

  // Strip duplicate opening if returned
  if (cleanCont.toLowerCase().startsWith(cleanOpening.toLowerCase())) {
    cleanCont = cleanCont.substring(cleanOpening.length).trim();
  }

  if (!cleanCont) return cleanOpening;

  if ([".", ",", "!", "?", ";", ":", "'", '"', "\n"].includes(cleanCont[0])) {
    return cleanOpening + cleanCont;
  }
  return cleanOpening + " " + cleanCont;
}

async function callGPT2(prompt, temperature, maxNewTokens, hfToken) {
  const headers = { "Content-Type": "application/json" };
  if (hfToken) {
    headers["Authorization"] = `Bearer ${hfToken}`;
  }

  const response = await fetch("https://api-inference.huggingface.co/models/openai-community/gpt2", {
    method: "POST",
    headers,
    body: JSON.stringify({
      inputs: prompt,
      parameters: {
        max_new_tokens: Math.min(Number(maxNewTokens) || 120, 250),
        temperature: Math.max(0.2, Math.min(Number(temperature) || 0.8, 1.8)),
        top_k: 50,
        top_p: 0.92,
        repetition_penalty: 1.1,
        return_full_text: false
      }
    })
  });

  if (!response.ok) {
    const errText = await response.text();
    throw new Error(`Hugging Face API error (${response.status}): ${errText}`);
  }

  const data = await response.json();
  if (Array.isArray(data) && data[0]?.generated_text) {
    return data[0].generated_text;
  } else if (data?.generated_text) {
    return data.generated_text;
  }
  return "";
}

module.exports = async function handler(req, res) {
  // Enable CORS
  res.setHeader("Access-Control-Allow-Credentials", true);
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "GET,OPTIONS,PATCH,DELETE,POST,PUT");
  res.setHeader("Access-Control-Allow-Headers", "X-CSRF-Token, X-Requested-With, Accept, Accept-Version, Content-Length, Content-MD5, Content-Type, Date, X-Api-Version");

  if (req.method === "OPTIONS") {
    res.status(200).end();
    return;
  }

  if (req.method !== "POST") {
    return res.status(405).json({ error: "Method not allowed. Use POST." });
  }

  try {
    const {
      opening_sentence,
      style = "Fantasy",
      temperature = 0.8,
      max_new_tokens = 120,
      is_comparison = false,
      temp_a = 0.5,
      temp_b = 1.2
    } = req.body || {};

    if (!opening_sentence || !opening_sentence.trim()) {
      return res.status(400).json({ error: "Opening sentence cannot be empty." });
    }

    const cleanOpening = opening_sentence.trim();
    const styleInstruction = STYLE_PROMPTS[style] || `Write a ${style.lower()} story.`;
    const prompt = `${styleInstruction} Continue the following opening sentence naturally and maintain a consistent narrative:\n\n${cleanOpening}`;
    const promptTokens = estimateTokenCount(prompt);
    const hfToken = process.env.HF_TOKEN || "";

    if (is_comparison) {
      // Comparison Mode
      const [contA, contB] = await Promise.all([
        callGPT2(prompt, temp_a, max_new_tokens, hfToken),
        callGPT2(prompt, temp_b, max_new_tokens, hfToken)
      ]);

      const storyA = formatStory(cleanOpening, contA);
      const storyB = formatStory(cleanOpening, contB);

      const newTokensA = estimateTokenCount(contA);
      const newTokensB = estimateTokenCount(contB);

      const divA = calculateLexicalDiversity(storyA);
      const divB = calculateLexicalDiversity(storyB);

      const summary = `### 🔬 Temperature Comparison Analysis\n\n` +
        `| Metric / Property | Output A (Temperature: ${temp_a}) | Output B (Temperature: ${temp_b}) |\n` +
        `| :--- | :--- | :--- |\n` +
        `| **Generated Tokens** | ~${newTokensA} tokens | ~${newTokensB} tokens |\n` +
        `| **Total Tokens** | ~${promptTokens + newTokensA} tokens | ~${promptTokens + newTokensB} tokens |\n` +
        `| **Lexical Diversity (TTR)** | ${divA} (unique / total) | ${divB} (unique / total) |\n` +
        `| **Relative Sampling Entropy** | ${temp_a <= temp_b ? "Lower" : "Higher"} (Sharper probabilities) | ${temp_b >= temp_a ? "Higher" : "Lower"} (Flatter probabilities) |\n` +
        `| **Primary Characteristic** | ${temp_a <= temp_b ? "More focused & predictable" : "More adventurous & varied"} | ${temp_b >= temp_a ? "More adventurous & varied" : "More focused & predictable"} |\n\n` +
        `#### 💡 Key Observed Tendencies:\n` +
        `1. **Coherence vs. Entropy:** At **T=${temp_a}**, sampling is ${temp_a < temp_b ? "more conservative and concentrated" : "more dispersed"}, whereas at **T=${temp_b}**, sampling is ${temp_b > temp_a ? "more exploratory and dispersed" : "more conservative"}.\n` +
        `2. **Vocabulary Variation:** Output A achieved a lexical diversity score of **${divA}**, whereas Output B achieved **${divB}**.\n` +
        `3. **Predictability vs. Divergence:** Lower temperatures preserve syntactic consistency, whereas elevated temperatures encourage creative vocabulary with an increased risk of narrative drift.`;

      return res.status(200).json({
        success: true,
        is_comparison: true,
        output_a: {
          story: storyA,
          temperature: Number(temp_a),
          new_tokens: newTokensA,
          total_tokens: promptTokens + newTokensA
        },
        output_b: {
          story: storyB,
          temperature: Number(temp_b),
          new_tokens: newTokensB,
          total_tokens: promptTokens + newTokensB
        },
        summary
      });
    }

    // Single Generation Mode
    const continuation = await callGPT2(prompt, temperature, max_new_tokens, hfToken);
    const story = formatStory(cleanOpening, continuation);
    const newTokens = estimateTokenCount(continuation);

    return res.status(200).json({
      success: true,
      story,
      style,
      temperature: Number(temperature),
      prompt_tokens: promptTokens,
      new_tokens: newTokens,
      total_tokens: promptTokens + newTokens,
      device: "HUGGINGFACE-SERVERLESS"
    });

  } catch (err) {
    console.error("Vercel Generation Error:", err);
    return res.status(500).json({
      success: false,
      error: err.message || "Failed to generate story."
    });
  }
};
