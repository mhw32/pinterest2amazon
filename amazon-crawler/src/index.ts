import _ from "lodash";
import puppeteer from "puppeteer";

const root = `https://www.amazon.com`;

// tslint:disable-next-line:interface-name
interface ProductInfo {
  src: string;
  link: string;
}

(async () => {
  const browser = await puppeteer.launch({ headless: false });
  const page = await browser.newPage();
  await page.goto(root);
  await page.setViewport({
    width: 2000,
    height: 1000
  });
  await page.type("#twotabsearchtextbox", "tennis rackets");
  await page.click("input.nav-input");
  await sleep(2000);
  await autoScroll(page);

  const carouselBoxes = await page.$$("li.a-carousel-card");
  const productBoxes = await page.$$("div.a-section");
  const results: ProductInfo[] = [];
  for (const box of [...carouselBoxes, ...productBoxes]) {
    const info = await processBox(box);
    if (info) {
      results.push(info);
    }
  }

  const uniqueResults = _.uniqBy(results, e => [e.src, e.link].join("--"));
  console.log(
    uniqueResults
      .map(({ src, link }) => `src=${src}\nlink=${link}`)
      .join("\n\n")
  );
  console.log(`found ${uniqueResults.length} images and links`);
  await browser.close();
})();

async function processBox(
  box: puppeteer.ElementHandle
): Promise<ProductInfo | null> {
  const img = await box.$("img.s-image");
  if (!img) {
    return null;
  }
  const link = await box.$("a.a-link-normal");
  if (!link) {
    return null;
  }
  const imageSrc = await (await img.getProperty("src")).jsonValue();
  const productLink = await (await link.getProperty("href")).jsonValue();
  return { src: imageSrc, link: productLink };
}

async function autoScroll(page: puppeteer.Page) {
  await page.evaluate(async () => {
    await new Promise((resolve, reject) => {
      let totalHeight = 0;
      const distance = 100;
      const timer = setInterval(() => {
        const scrollHeight = document.body.scrollHeight;
        window.scrollBy(0, distance);
        totalHeight += distance;
        if (totalHeight >= scrollHeight) {
          clearInterval(timer);
          resolve();
        }
      }, 100);
    });
  });
}

function sleep(time: number) {
  return new Promise(resolve => {
    setTimeout(resolve, time);
  });
}
