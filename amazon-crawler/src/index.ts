import puppeteer from "puppeteer";

const root = `https://www.amazon.com`;

function sleep(time: number) {
  return new Promise(function(resolve) {
    setTimeout(resolve, time)
  });
}

(async () => {
  const browser = await puppeteer.launch({ headless: false });
  const page = await browser.newPage();
  await page.goto('https://www.amazon.com');
  await page.type('#twotabsearchtextbox', 'tennis rackets');
  await page.click('input.nav-input');
  await sleep(2000);
  const productBoxes = await page.$$('li.a-carousel-card');
  const results = [];
  for (const box of productBoxes){
    // anchors =>  anchors.map(anchor => anchor.getAttribute('href'))
    const link = await box.$('a.a-link-normal');
    if (!link) {

    }
    await link();
    const img = await box.$('img.s-img');
  }

  console.log(productBoxes.map(e => e ? `${root}${e.trim()}` : null).filter(e => e));
  console.log(productBoxes.length);
  await browser.close();
})();
