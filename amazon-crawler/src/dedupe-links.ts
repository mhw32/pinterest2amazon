import axios from "axios";
import fs from "fs";

(async () => {
  const buffer = await fs.readFileSync("furniture.txt");
  const rows = buffer.toString().split("\n");
  console.log(`found ${rows.length} rows`);
  const foundLinks = new Set<string>();
  for (const row of rows) {
    const [imgSrc, link] = row.split(",");
    foundLinks.add(link);
  }
  console.log(`${[...foundLinks].length} links after deduping`);

  const foundImages = new Set<string>();
  for (const [i, row] of rows.entries()) {
    const [imgSrc] = row.split(",");
    try {
      const image = await axios.get(imgSrc, { responseType: "arraybuffer" });
      const data = Buffer.from(image.data).toString("base64");
      foundImages.add(data);
    } catch (err) {
      console.error(`failed to get data for ${imgSrc}`, err.message);
    }
    if (i % 100 === 0) {
      console.log(`processed ${i} image links`);
    }
  }
  console.log(`${[...foundImages].length} images after deduping`);
})();
