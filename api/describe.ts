// api/describe/route.ts
export const runtime = 'nodejs';          // use Node.js, not Edge
export const dynamic = 'force-dynamic';   // ensure server execution

import OpenAI from 'openai';

const getOpenAIClient = () => {
  if (!process.env.OPENAI_API_KEY) {
    throw new Error('OPENAI_API_KEY environment variable is required');
  }

  return new OpenAI({
    apiKey: process.env.OPENAI_API_KEY,
  });
};

export async function generateImageDescription(base64Image: string, mimeType: string): Promise<string> {
  const openai = getOpenAIClient();

  const response = await openai.chat.completions.create({
    model: 'gpt-5-nano',
    messages: [
      {
        role: 'user',
        content: [
          {
            type: 'text',
            text: 'You create descriptions of images. Provided with an image describe the image. You can describe unambiguously the image',
          },
          {
            type: 'image_url',
            image_url: {
              url: `data:${mimeType};base64,${base64Image}`,
            },
          },
        ],
      },
    ],
  });

  const caption = response.choices[0]?.message?.content;

  if (!caption) {
    throw new Error('Failed to generate caption');
  }

  return caption;
}

async function convertFileToBase64(file: File): Promise<{ base64: string; mimeType: string }> {
  const bytes = await file.arrayBuffer();
  const buffer = Buffer.from(bytes);
  const base64 = buffer.toString('base64');
  const mimeType = file.type || 'image/jpeg';

  return { base64, mimeType };
}

export async function POST(request: Request) {
  try {
    const formData = await request.formData();
    const file = formData.get('image') as File;

    if (!file) {
      return new Response(
        JSON.stringify({ error: 'No image file provided' }),
        { status: 400, headers: { 'content-type': 'application/json' } }
      );
    }

    const { base64, mimeType } = await convertFileToBase64(file);
    const caption = await generateImageDescription(base64, mimeType);

    return new Response(JSON.stringify(caption), {
        status: 200,
        headers: { 'content-type': 'application/json' },
    });
  } catch (error) {
    console.error('Error generating caption:', error);
    return new Response(
        JSON.stringify({ error: 'Failed to process image' }),
        { status: 500, headers: { 'content-type': 'application/json' } }
    );
  }
}
