import { InferenceClient } from "@huggingface/inference";
import dotenv from "dotenv";
import fs from "fs/promises";

dotenv.config();

// Debug: Log token (remove in production)
console.log("HF_ACCESS_TOKEN:", process.env.HF_ACCESS_TOKEN ? "Set (starts with hf_)" : "UNDEFINED - Check .env!");

const hf = new InferenceClient(process.env.HF_ACCESS_TOKEN);

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
  // Text Classification
  await runTask("Text Classification", () =>
    hf.textClassification({
      model: "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
      inputs: "I love Hugging Face!",
    })
  );

  // Text to Image
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

  // Chat Completion
  await runTask("Chat Completion", () =>
    hf.chatCompletion({
      model: "meta-llama/Meta-Llama-3-8B-Instruct",
      messages: [{ role: "user", content: "Hello!" }],
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

  // Image Segmentation
  await runTask("Image Segmentation", () =>
    hf.imageSegmentation({
      model: "facebook/detr-resnet-50-panoptic",
      inputs: "https://upload.wikimedia.org/wikipedia/commons/4/47/American_Eskimo_Dog.jpg",
    })
  );

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
}

runAllTasks().catch((error) => console.error("Global Error:", error.message));