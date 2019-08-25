import axios from "axios";
import fs from "fs";
import Path from "path";

(async () => {
  const buffer = await fs.readFileSync("csvs/women\'s+shoes-dedupe.txt");
  const rows = buffer.toString().split("\n");
  const imageLinkMap: {
    [filename: string]: { src: string; link: string };
  } = {};
  for (const row of rows) {
    const [src, link] = row.split(",");
    const filename = `${random()}.jpg`;
    await downloadImage(src, filename);
    imageLinkMap[filename] = {
      src,
      link
    };
  }
  await fs.writeFileSync(
    "image-map.json",
    JSON.stringify(imageLinkMap, null, 2)
  );
})();

async function downloadImage(url: string, filename: string) {
  const path = Path.resolve(__dirname, "train", filename);
  const writer = fs.createWriteStream(path);

  const response = await axios({
    url,
    method: "GET",
    responseType: "stream"
  });

  response.data.pipe(writer);

  return new Promise((resolve, reject) => {
    writer.on("finish", resolve);
    writer.on("error", reject);
  });
}

function random() {
  return (
    Math.random()
      .toString(36)
      .substring(2, 15) +
    Math.random()
      .toString(36)
      .substring(2, 15)
  );
}
