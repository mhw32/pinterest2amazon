import axios from "axios";
import fs from "fs";

interface ProductInfo {
  src: string;
  link: string;
}

(async () => {
  const buffer = await fs.readFileSync("furniture.txt");
  const rows = buffer.toString().split("\n");
  console.log(`found ${rows.length} rows`);
  const foundLinks = new Set<string>();
  const foundImages = new Set<string>();
  const uniqueProducts: ProductInfo[] = [];
  for (const [i, row] of rows.entries()) {
    const [imgSrc, link] = row.split(",");
    let data: string | null = null;
    try {
      const image = await axios.get(imgSrc, { responseType: "arraybuffer" });
      data = Buffer.from(image.data).toString("base64");
    } catch (err) {
      console.error(`failed to get data for ${imgSrc}`, err.message);
    }
    if (i % 100 === 0) {
      console.log(`processed ${i} image links`);
    }
    if (data) {
      // check if should add to uniqueProducts
      const isDupe = foundImages.has(data) || foundLinks.has(link);
      if (!isDupe) {
        uniqueProducts.push({ link, src: imgSrc });
      }
      // add values to set
      foundImages.add(data);
      foundLinks.add(link);
    }
  }
  console.log(`${uniqueProducts.length} uniq products after deduping`);
  await fs.writeFileSync(
    "furniture-dedupe.txt",
    uniqueProducts.map(({ src, link }) => [src, link].join(",")).join("\n")
  );
})();
