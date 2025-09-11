import { InferenceClient } from "@huggingface/inference";
import dotenv from "dotenv";
import fs from "fs/promises";
import path from "path";
import fetch from "node-fetch"; // Add: npm install node-fetch

dotenv.config();

// Debug: Log token (remove in production)
console.log("HF_ACCESS_TOKEN:", process.env.HF_ACCESS_TOKEN ? "Set (starts with hf_)" : "UNDEFINED - Check .env!");

const hf = new InferenceClient(process.env.HF_ACCESS_TOKEN); // Direct string - fixes startsWith error

// Helper to fetch image as ArrayBuffer
async function fetchImageAsBuffer(url) {
  try {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`Failed to fetch image: ${response.status}`);
    return await response.arrayBuffer();
  } catch (error) {
    console.error("Fetch Image Error:", error.message);
    return null;
  }
}

async function runTask(taskName, taskFn) {
  try {
    const result = await taskFn();
    console.log(`${taskName}:`, result);
    return result;
  } catch (error) {
    console.error(`${taskName} Error:`, error.message);
    if (error.httpResponse) {
      console.error("HTTP Response:", error.httpResponse.body?.detail || error.httpResponse);
    }
    return null;
  }
}

async function runAllTasks() {
  // Text Classification (supported on hf-inference)
  await runTask("Text Classification", () =>
    hf.textClassification({
      model: "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
      inputs: "I love Hugging Face!",
    })
  );

  // Text to Image (use fal-ai provider for stability)
  const textToImageResult = await runTask("Text to Image", () =>
    hf.textToImage({
      model: "stabilityai/stable-diffusion-3-medium",
      inputs: "A cute cat wearing a spacesuit, high quality, vibrant colors",
      parameters: { negative_prompt: "blurry, low quality", num_inference_steps: 28 },
    })
  );
  if (textToImageResult) {
    await fs.writeFile("output.png", Buffer.from(await textToImageResult.arrayBuffer()));
    console.log("Text to Image: Saved to output.png");
  }


  // Automatic Speech Recognition (use hf-inference supported model)
  await runTask("Automatic Speech Recognition", () =>
    hf.automaticSpeechRecognition({
      model: "openai/whisper-large-v3-turbo",
      inputs: "https://samplelib.com/lib/preview/mp3/sample-15s.mp3",
    })
  );

  // Chat Completion (use novita or auto; ensure token access for LLaMA)
  await runTask("Chat Completion", () =>
    hf.chatCompletion({
      model: "meta-llama/Meta-Llama-3-8B-Instruct",
      messages: [{ role: "user", content: "Hello!" }],
    })
  );

  // Document Question Answering - Use alternative supported model
  await runTask("Document Question Answering", () =>
    hf.documentQuestionAnswering({
      model: "microsoft/layoutlmv2-base-uncased",
      inputs: {
        image: "https://www.google.com/imgres?q=today%20date&imgurl=https%3A%2F%2Fwww.inchcalculator.com%2Fwp-content%2Fuploads%2F2023%2F11%2Fcurrent-date-formula.png&imgrefurl=https%3A%2F%2Fwww.kalagadget.com%2F%3Fq%3D69421728051790%26mod%3D3d7c6263%26uri%3Dpin.php%253Fid%253D2321990-553%2526name%253Ddate%2Btoday&docid=wXsWJtYGvDG8aM&tbnid=b18RcWJIIcc8oM&vet=12ahUKEwji35eLy9GPAxUt2DgGHX76EoYQM3oECBkQAA..i&w=1280&h=854&hcb=2&itg=1&ved=2ahUKEwji35eLy9GPAxUt2DgGHX76EoYQM3oECBkQAA",
        question: "What is the invoice date?",
      },
    })
  );

  // Feature Extraction
  await runTask("Feature Extraction", () =>
    hf.featureExtraction({
      model: "sentence-transformers/all-MiniLM-L6-v2",
      inputs: "This is a test sentence.",
    })
  );

  // Fill Mask
  await runTask("Fill Mask", () =>
    hf.fillMask({
      model: "google-bert/bert-base-uncased",
      inputs: "The capital of France is [MASK].",
    })
  );

  // Image Classification (fetch buffer for safety)
  const imgBuffer = await fetchImageAsBuffer("https://upload.wikimedia.org/wikipedia/commons/4/47/American_Eskimo_Dog.jpg");
  if (imgBuffer) {
    await runTask("Image Classification", () =>
      hf.imageClassification({
        model: "google/vit-base-patch16-224-in21k",
        inputs: imgBuffer, // Use buffer instead of URL
      })
    );
  }

  // Image Segmentation (use buffer)
  if (imgBuffer) {
    await runTask("Image Segmentation", () =>
      hf.imageSegmentation({
        model: "facebook/detr-resnet-50-panoptic",
        inputs: imgBuffer,
      })
    );
  }

  // Image to Image (use supported model with provider)
  if (imgBuffer) {
    const imgToImgResult = await runTask("Image to Image", () =>
      hf.imageToImage({
        model: "stabilityai/stable-diffusion-xl-base-1.0",
        inputs: {
          image: imgBuffer,
          prompt: "Enhance depth, realistic style",
        },
      })
    );
    if (imgToImgResult) {
      await fs.writeFile("img_to_img.png", Buffer.from(await imgToImgResult.arrayBuffer()));
      console.log("Image to Image: Saved to img_to_img.png");
    }
  }

  // Image to Text (use supported model)
  await runTask("Image to Text", () =>
    hf.imageToText({
      model: "nlpconnect/vit-gpt2-image-captioning",
      inputs: "https://upload.wikimedia.org/wikipedia/commons/4/47/American_Eskimo_Dog.jpg", // URL works here
    })
  );


  // Object Detection (use buffer)
  if (imgBuffer) {
    await runTask("Object Detection", () =>
      hf.objectDetection({
        model: "facebook/detr-resnet-50",
        inputs: imgBuffer,
      })
    );
  }

  // Question Answering
  await runTask("Question Answering", () =>
    hf.questionAnswering({
      model: "deepset/roberta-base-squad2",
      inputs: { question: "What is Hugging Face?", context: "Hugging Face is an AI company." },
    })
  );

  // Sentence Similarity
  await runTask("Sentence Similarity", () =>
    hf.sentenceSimilarity({
      model: "sentence-transformers/all-mpnet-base-v2",
      inputs: { source_sentence: "Hello world", sentences: ["Hi universe", "Goodbye"] },
    })
  );

  // Summarization
  await runTask("Summarization", () =>
    hf.summarization({
      model: "facebook/bart-large-cnn",
      inputs:
        "Hugging Face is a company focused on advancing AI technologies, particularly in natural language processing and computer vision. They provide open-source tools and models for developers.",
    })
  );

  // Table Question Answering
  await runTask("Table Question Answering", () =>
    hf.tableQuestionAnswering({
      model: "google/tapas-base-finetuned-wtq",
      inputs: {
        query: "What is the capital of France?",
        table: {
          Country: ["France", "Germany"],
          Capital: ["Paris", "Berlin"],
        },
      },
    })
  );


  // Text Generation (use supported model)
  await runTask("Text Generation", () =>
    hf.textGeneration({
      model: "microsoft/DialoGPT-medium",
      inputs: "Once upon a time",
    })
  );

  // Text to Speech (use supported model)
  const ttsResult = await runTask("Text to Speech", () =>
    hf.textToSpeech({
      model: "microsoft/speecht5_tts",
      inputs: "Hello, world!",
    })
  );
  if (ttsResult) {
    await fs.writeFile("tts.wav", Buffer.from(await ttsResult.arrayBuffer()));
    console.log("Text to Speech: Saved to tts.wav");
  }


  // Token Classification
  await runTask("Token Classification", () =>
    hf.tokenClassification({
      model: "dbmdz/bert-large-cased-finetuned-conll03-english",
      inputs: "Hugging Face is in New York.",
    })
  );

  // Translation
  await runTask("Translation", () =>
    hf.translation({
      model: "Helsinki-NLP/opus-mt-en-fr",
      inputs: "Hello, how are you?",
    })
  );

  // Visual Question Answering - Use alternative
  await runTask("Visual Question Answering", () =>
    hf.visualQuestionAnswering({
      model: "dandelin/vilt-b32-finetuned-vqa", // Try; if fails, use "nlpconnect/vit-gpt2-image-captioning" as fallback
      inputs: {
        image: "https://upload.wikimedia.org/wikipedia/commons/4/47/American_Eskimo_Dog.jpg",
        question: "What color is it?",
      },
    })
  );

  // Zero Shot Classification
  await runTask("Zero Shot Classification", () =>
    hf.zeroShotClassification({
      model: "facebook/bart-large-mnli",
      inputs: "This is a test.",
      parameters: { candidate_labels: ["positive", "negative"] },
    })
  );

  // Zero Shot Image Classification (use supported model)
  await runTask("Zero Shot Image Classification", () =>
    hf.zeroShotImageClassification({
      model: "openai/clip-vit-base-patch32", // Smaller, supported variant
      inputs: {
        image: "https://upload.wikimedia.org/wikipedia/commons/4/47/American_Eskimo_Dog.jpg",
        candidate_labels: ["cat", "dog"],
      },
    })
  );
}

runAllTasks().catch((error) => console.error("Global Error:", error.message));