import express from "express";
import path from "path";
import dotenv from "dotenv";
import { GoogleGenAI, Type } from "@google/genai";
import { createServer as createViteServer } from "vite";

dotenv.config();

const app = express();
const PORT = 3000;

// Body parser
app.use(express.json({ limit: "50mb" }));
app.use(express.urlencoded({ extended: true, limit: "50mb" }));

// Server-side Gemini initialization
let aiClient: GoogleGenAI | null = null;
function getGeminiClient(): GoogleGenAI {
  if (!aiClient) {
    aiClient = new GoogleGenAI({
      apiKey: process.env.GEMINI_API_KEY,
      httpOptions: {
        headers: {
          "User-Agent": "aistudio-build",
        },
      },
    });
  }
  return aiClient;
}

// Health check
app.get("/api/health", (req, res) => {
  res.json({
    status: "ok",
    timestamp: new Date().toISOString(),
    service: "NutriSilage AI Engine",
    blockchainNode: "NDDB-AgriLedger-Mainnet-Node-04",
  });
});

// AI Feed Analysis Endpoint
app.post("/api/analyze-feed", async (req, res) => {
  try {
    const { imageBase64, mimeType, feedType, sensoryInputs, farmerNotes } = req.body;

    const hasImage = Boolean(imageBase64);
    const client = getGeminiClient();

    const systemPrompt = `You are the Chief Dairy Nutritionist and Computer Vision Silage Quality Specialist at the Indian Council of Agricultural Research (ICAR-NDRI) and National Dairy Development Board (NDDB).
Your task is to perform rapid, rigorous nutritional, microbiological, and physical quality estimation of dairy feed / silage (e.g. Corn/Maize silage, Sorghum, Hybrid Napier, Alfalfa/Lucerne, TMR, Paddy/Wheat Straw) for Indian dairy farmers.

Analyze the visual evidence (if provided) and manual sensory inputs (color, odor, moisture feel, fermentation, mold, particle distribution) and return a complete JSON response according to the schema.
Ensure calculations are realistic for dairy ruminant nutrition:
- Crude Protein (CP % DM): typical corn silage 7.5-9.5%, Lucerne 17-22%, Napier 8-12%, Straw 3-4.5%.
- Dry Matter (DM %): optimal silage 30-38% (Moisture 62-70%). If moisture >75%, high butyric acid risk. If <55%, poor packing / aerobic mold risk.
- pH: optimal corn silage 3.7 - 4.2; grass silage 4.2 - 4.8. pH > 5.0 indicates clostridial/secondary spoilage.
- Acid Detergent Fiber (ADF % DM): 22-30% (lower is better digestibility).
- Neutral Detergent Fiber (NDF % DM): 38-50% (optimal rumen fill).
- Total Digestible Nutrients (TDN % DM): 62-72% for top silage.
- Digestibility Rating (NDFD 48hr %): 50-70%.
- Lactic / Acetic / Butyric acid estimation.
- Mold & Mycotoxin (Aflatoxin, Deoxynivalenol, Zearalenone) risk index (Low, Moderate, Critical).
- Penn State Particle Size breakdown percentages (>19mm top screen, 8-19mm middle, 1.18-8mm lower, <1.18mm bottom pan).
- Projected Milk Yield Impact (e.g. +1.5 L/cow/day or -2.0 L/cow/day) and Fat % impact.
- Actionable Veterinary advice for Indian dairy farmers (ration balancing, inoculant use, sealing, toxin binder, buffer inclusion like sodium bicarbonate).`;

    let userPromptText = `Please analyze this dairy feed/silage sample.
Feed Type: ${feedType || "Corn Silage"}
Sensory Inputs:
- Color: ${sensoryInputs?.color || "Not specified"}
- Odor / Smell: ${sensoryInputs?.odor || "Not specified"}
- Moisture Squeeze: ${sensoryInputs?.moistureFeel || "Not specified"}
- Mold Observation: ${sensoryInputs?.moldPresence || "Not observed"}
- Storage / Ensilage Days: ${sensoryInputs?.storageDays || "45"} days
- Storage Method: ${sensoryInputs?.storageType || "Bunker Silo / Silage Pit"}
Additional Notes: ${farmerNotes || "None"}
`;

    let contentsPayload: any;

    if (hasImage) {
      // Strip potential data URL prefix
      const cleanBase64 = imageBase64.replace(/^data:image\/[a-z]+;base64,/, "");
      contentsPayload = {
        parts: [
          {
            inlineData: {
              mimeType: mimeType || "image/jpeg",
              data: cleanBase64,
            },
          },
          {
            text: userPromptText + "\nInspect the image closely for grain content, chop length uniformity, compaction, mold patches (white/blue/black), moisture sheen, and discoloration.",
          },
        ],
      };
    } else {
      contentsPayload = userPromptText;
    }

    const response = await client.models.generateContent({
      model: "gemini-3.7-flash",
      contents: contentsPayload,
      config: {
        systemInstruction: systemPrompt,
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            overallScore: {
              type: Type.NUMBER,
              description: "Quality score from 0 to 100",
            },
            grade: {
              type: Type.STRING,
              description: "Grade: 'A+ Supreme', 'A Prime', 'B Good', 'C Fair', or 'D Hazardous'",
            },
            summary: {
              type: Type.STRING,
              description: "Executive summary of feed quality and suitability for high-yielding dairy cattle",
            },
            nutritionalParameters: {
              type: Type.OBJECT,
              properties: {
                crudeProteinPercent: { type: Type.NUMBER, description: "Crude Protein % DM" },
                dryMatterPercent: { type: Type.NUMBER, description: "Dry Matter %" },
                moisturePercent: { type: Type.NUMBER, description: "Moisture %" },
                phLevel: { type: Type.NUMBER, description: "Silage pH value" },
                adfPercent: { type: Type.NUMBER, description: "Acid Detergent Fiber %" },
                ndfPercent: { type: Type.NUMBER, description: "Neutral Detergent Fiber %" },
                tdnPercent: { type: Type.NUMBER, description: "Total Digestible Nutrients %" },
                metabolizableEnergyMjPerKg: { type: Type.NUMBER, description: "ME (MJ/kg DM)" },
                digestibilityIndex: { type: Type.NUMBER, description: "Digestibility % (0-100)" },
              },
              required: [
                "crudeProteinPercent",
                "dryMatterPercent",
                "moisturePercent",
                "phLevel",
                "adfPercent",
                "ndfPercent",
                "tdnPercent",
                "metabolizableEnergyMjPerKg",
                "digestibilityIndex",
              ],
            },
            fermentationProfile: {
              type: Type.OBJECT,
              properties: {
                lacticAcidPercent: { type: Type.NUMBER, description: "Lactic acid % (ideally 4-7%)" },
                aceticAcidPercent: { type: Type.NUMBER, description: "Acetic acid % (ideally 1-3%)" },
                butyricAcidPercent: { type: Type.NUMBER, description: "Butyric acid % (ideally <0.1%)" },
                ammoniaNitrogenRatio: { type: Type.NUMBER, description: "NH3-N as % of Total N" },
                fermentationQuality: { type: Type.STRING, description: "Excellent, Desirable, Secondary Spoilage, Clostridial" },
              },
              required: [
                "lacticAcidPercent",
                "aceticAcidPercent",
                "butyricAcidPercent",
                "ammoniaNitrogenRatio",
                "fermentationQuality",
              ],
            },
            microbiologicalAndPhysical: {
              type: Type.OBJECT,
              properties: {
                moldRiskLevel: { type: Type.STRING, description: "None/Negligible, Low, Moderate, High, Critical" },
                mycotoxinThreat: { type: Type.STRING, description: "Aflatoxin / DON threat assessment" },
                particleSizingPSPS: {
                  type: Type.OBJECT,
                  properties: {
                    topScreenOver19mm: { type: Type.NUMBER, description: "Top sieve % (>19mm)" },
                    middleScreen8to19mm: { type: Type.NUMBER, description: "Middle sieve % (8-19mm)" },
                    lowerScreen1to8mm: { type: Type.NUMBER, description: "Lower sieve % (1.18-8mm)" },
                    bottomPanUnder1mm: { type: Type.NUMBER, description: "Bottom pan % (<1.18mm)" },
                  },
                  required: ["topScreenOver19mm", "middleScreen8to19mm", "lowerScreen1to8mm", "bottomPanUnder1mm"],
                },
                foreignMaterialRisk: { type: Type.STRING, description: "Dirt, rocks, dead matter risk assessment" },
                packingDensityScore: { type: Type.NUMBER, description: "Compaction / density score 0-100" },
              },
              required: ["moldRiskLevel", "mycotoxinThreat", "particleSizingPSPS", "foreignMaterialRisk", "packingDensityScore"],
            },
            dairyOutcomeProjections: {
              type: Type.OBJECT,
              properties: {
                projectedMilkYieldChangeLitersPerDay: {
                  type: Type.NUMBER,
                  description: "Expected daily milk yield difference per cow in Liters (e.g. +1.8 or -1.5)",
                },
                fatPercentageImpact: {
                  type: Type.NUMBER,
                  description: "Expected milk fat % difference (e.g. +0.2 or -0.3)",
                },
                rumenAcidosisRisk: { type: Type.STRING, description: "Low, Moderate, High" },
                estimatedFeedCostEfficiencyPercent: { type: Type.NUMBER, description: "Feed cost efficiency index 0-100" },
              },
              required: [
                "projectedMilkYieldChangeLitersPerDay",
                "fatPercentageImpact",
                "rumenAcidosisRisk",
                "estimatedFeedCostEfficiencyPercent",
              ],
            },
            arOverlayData: {
              type: Type.OBJECT,
              properties: {
                detectedFeatures: {
                  type: Type.ARRAY,
                  items: {
                    type: Type.OBJECT,
                    properties: {
                      label: { type: Type.STRING },
                      confidence: { type: Type.NUMBER },
                      status: { type: Type.STRING, description: "optimal, warning, hazard, info" },
                      description: { type: Type.STRING },
                      boundingBox: {
                        type: Type.OBJECT,
                        properties: {
                          x: { type: Type.NUMBER },
                          y: { type: Type.NUMBER },
                          width: { type: Type.NUMBER },
                          height: { type: Type.NUMBER },
                        },
                        required: ["x", "y", "width", "height"],
                      },
                    },
                    required: ["label", "confidence", "status", "description", "boundingBox"],
                  },
                },
                colorSpectrumAnalysis: {
                  type: Type.STRING,
                  description: "Explanation of color chromatography (e.g. 74% Olive Green, 18% Amber, 8% Pale)",
                },
                moistureDensityHeatmapScore: { type: Type.NUMBER },
              },
              required: ["detectedFeatures", "colorSpectrumAnalysis", "moistureDensityHeatmapScore"],
            },
            actionableRecommendations: {
              type: Type.ARRAY,
              items: { type: Type.STRING },
              description: "5 practical step-by-step instructions for the dairy farmer",
            },
            laboratoryVerificationRecommendation: {
              type: Type.OBJECT,
              properties: {
                isLabTestUrgent: { type: Type.BOOLEAN },
                reason: { type: Type.STRING },
                recommendedTests: {
                  type: Type.ARRAY,
                  items: { type: Type.STRING },
                },
                confidenceScore: { type: Type.NUMBER, description: "AI prediction confidence percentage 0-100" },
              },
              required: ["isLabTestUrgent", "reason", "recommendedTests", "confidenceScore"],
            },
          },
          required: [
            "overallScore",
            "grade",
            "summary",
            "nutritionalParameters",
            "fermentationProfile",
            "microbiologicalAndPhysical",
            "dairyOutcomeProjections",
            "arOverlayData",
            "actionableRecommendations",
            "laboratoryVerificationRecommendation",
          ],
        },
      },
    });

    const parsedResult = JSON.parse(response.text || "{}");
    res.json({ success: true, data: parsedResult });
  } catch (error: any) {
    console.error("AI Analysis Error:", error);
    res.status(500).json({
      success: false,
      error: error.message || "Failed to analyze feed sample",
    });
  }
});

// AI Dairy Copilot Assistant Endpoint
app.post("/api/ai-copilot", async (req, res) => {
  try {
    const { message, language, farmContext, conversationHistory } = req.body;

    const client = getGeminiClient();

    const systemPrompt = `You are 'Kisan Dairy AI', the intelligent dairy nutritionist & silage engineering copilot for Indian dairy farmers.
You specialize in:
1. Silage making, bunker pit sealing, inoculant bacteria (L. plantarum / L. buchneri), moisture determination via microwave test.
2. Balancing Total Mixed Ration (TMR) with green fodder (Maize, Napier, SSG, Lucerne) + dry roughage (Paddy/Wheat Straw) + concentrate feeds (Mustard cake, cotton seed, bypass fat, mineral mixture).
3. Troubleshooting feed quality: spoilage, clostridial smell (butyric), mold removal, mycotoxin mitigation with toxin binders (HSCAS, yeast cell walls).
4. Improving milk yield, fat %, SNF, preventing Subacute Ruminal Acidosis (SARA), ketosis, and mastitis.
5. Dairy breeds in India: Murrah/Jaffarabadi Buffaloes, Gir, Sahiwal, Red Sindhi, HF crossbreds, Jersey crossbreds.

Current preferred language: ${language || "English"}.
If language is 'Telugu', respond in Telugu (తెలుగు) with clear agricultural terminology.
If language is 'Hindi', respond in Hindi (हिंदी).
If 'English', respond in clear, helpful English.
Keep answers actionable, structured with bullet points, empathetic to smallholder farmers, and scientifically sound.`;

    const chatHistory = (conversationHistory || []).map((msg: any) => ({
      role: msg.role === "assistant" ? "model" : "user",
      parts: [{ text: msg.content }],
    }));

    const response = await client.models.generateContent({
      model: "gemini-3.7-flash",
      contents: [
        ...chatHistory,
        {
          role: "user",
          parts: [
            {
              text: `${message}\nFarm Context: Cattle count: ${farmContext?.cattleCount || 12}, Main fodder: ${farmContext?.mainFodder || "Corn Silage"}, Cooperative: ${farmContext?.cooperative || "Local Milk Union"}.`,
            },
          ],
        },
      ],
      config: {
        systemInstruction: systemPrompt,
      },
    });

    res.json({
      success: true,
      reply: response.text || "I am analyzing your dairy feeding query...",
    });
  } catch (error: any) {
    console.error("AI Copilot Error:", error);
    res.status(500).json({
      success: false,
      error: error.message || "Failed to generate copilot response",
    });
  }
});

// Blockchain Verification Mock Ledger Endpoint
app.post("/api/verify-certificate", async (req, res) => {
  try {
    const { certHash, batchId } = req.body;
    
    // Simulate real cryptographically verified query on the AgriLedger
    const simulatedBlock = {
      verified: true,
      certHash: certHash || "0x8f73b145a2789cd6493b8214ee17649d854e4c2784530018fa5e26b1",
      batchId: batchId || "BATCH-SILAGE-2026-IND-082",
      blockHeight: 18492041,
      timestamp: new Date().toISOString(),
      network: "AgriLedger NDDB-EVM PoA (Proof of Authority)",
      consensusValidator: "ICAR-NDRI Karnal Node #07",
      ipfsCid: `ipfs://bafybeic${Math.random().toString(36).substring(2, 10)}silagefeedcert`,
      issuer: "NutriSilage AI Certified Autonomous Oracle v3.2",
      smartContract: "0x39218F504A7B132c324B32c1EaB918DeC7a18f40",
      status: "VALID_IMMUTABLE",
    };

    res.json({ success: true, data: simulatedBlock });
  } catch (err: any) {
    res.status(500).json({ success: false, error: err.message });
  }
});

// Setup Vite development middleware or production static serving
async function startServer() {
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), "dist");
    app.use(express.static(distPath));
    app.get("*", (req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`🌾 NutriSilage AI Engine active on http://0.0.0.0:${PORT}`);
  });
}

startServer();
