/* Fly Brain Blackjack - 3D table view.
   Polls /state from table_server.py and draws the room, the table, the flies,
   their cards and chips, and a live replay of the deciding fly's brain. */

(() => {
  const POLL_MS = 220;
  const BRAIN_PLAY_MS = 1500;
  const SEAT_ANGLES = [-62, -24, 62, 22];   // degrees around the table, viewer side; last = human
  const SEAT_RADIUS = 3.2, CARD_RADIUS = 1.8, CHIP_RADIUS = 2.55;
  const START_STACK = 20;

  // ---------- helpers ---------------------------------------------------
  const easeOut = k => 1 - Math.pow(1 - k, 3);
  const easeInOut = k => k < .5 ? 4 * k * k * k : 1 - Math.pow(-2 * k + 2, 3) / 2;
  const tweens = [];
  function tween(duration, fn, { ease = easeOut, delay = 0, done } = {}) {
    tweens.push({ t0: performance.now() + delay, duration, fn, ease, done });
  }
  function runTweens(now) {
    for (let i = tweens.length - 1; i >= 0; i--) {
      const tw = tweens[i];
      if (now < tw.t0) continue;
      const k = Math.min(1, (now - tw.t0) / tw.duration);
      tw.fn(tw.ease(k), k);
      if (k >= 1) { tweens.splice(i, 1); tw.done && tw.done(); }
    }
  }
  const srgb = (tex) => { tex.encoding = THREE.sRGBEncoding; return tex; };
  const canvasTex = (w, h, draw) => {
    const c = document.createElement("canvas"); c.width = w; c.height = h;
    draw(c.getContext("2d"), w, h);
    const t = new THREE.CanvasTexture(c); t.anisotropy = 8; return t;
  };

  // ---------- renderer / scene ------------------------------------------
  const container = document.getElementById("scene");
  const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 1.75));
  renderer.setSize(innerWidth, innerHeight);
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.05;
  container.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x07080c);
  scene.fog = new THREE.FogExp2(0x07080c, 0.045);

  const camera = new THREE.PerspectiveCamera(46, innerWidth / innerHeight, 0.1, 80);
  camera.position.set(0.6, 4.4, 6.9);

  const controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.target.set(0, 0.25, 0);
  controls.enableDamping = true; controls.dampingFactor = 0.06;
  controls.maxPolarAngle = Math.PI * 0.47; controls.minPolarAngle = Math.PI * 0.02;
  controls.minDistance = 3.2; controls.maxDistance = 11;
  controls.autoRotate = true; controls.autoRotateSpeed = 0.35;
  let lastInteraction = 0;
  renderer.domElement.addEventListener("pointerdown", () => { controls.autoRotate = false; lastInteraction = performance.now(); });
  renderer.domElement.addEventListener("wheel", () => { controls.autoRotate = false; lastInteraction = performance.now(); }, { passive: true });

  // camera follow: a view is a (position, target) pair the camera glides
  // towards whenever the viewer has not touched the controls for a while
  const OVERVIEW = { pos: new THREE.Vector3(0.6, 4.4, 6.9), target: new THREE.Vector3(0, 0.25, 0) };
  // the dealer's turn is watched from straight above, the whole table in frame
  const DEALER_VIEW = { pos: new THREE.Vector3(0, 8.2, 0.9), target: new THREE.Vector3(0, 0, -0.2) };
  let follow = true, view = OVERVIEW, viewKey = "overview";
  function seatView(angle) {
    return { pos: new THREE.Vector3(Math.sin(angle) * 5.9, 3.7, Math.cos(angle) * 5.9),
             target: new THREE.Vector3(Math.sin(angle) * 0.4, 0.1, Math.cos(angle) * 0.4) };
  }
  function setView(key, v) { if (viewKey !== key) { viewKey = key; view = v; } }
  const camEl = document.getElementById("cam");
  camEl.addEventListener("click", () => {
    follow = !follow; camEl.classList.toggle("on", follow);
    camEl.querySelector("b").textContent = follow ? "עוקבת" : "חופשית";
    if (!follow) controls.autoRotate = true;
  });

  // post-processing: bloom for the eyes, chips and thought bubbles
  const composer = new THREE.EffectComposer(renderer);
  composer.addPass(new THREE.RenderPass(scene, camera));
  const bloom = new THREE.UnrealBloomPass(new THREE.Vector2(innerWidth, innerHeight), 0.55, 0.6, 0.86);
  composer.addPass(bloom);
  composer.addPass(new THREE.ShaderPass(THREE.GammaCorrectionShader));

  // ---------- lights ----------------------------------------------------
  scene.add(new THREE.HemisphereLight(0x8090c0, 0x2a1a0c, 0.35));
  const lamp = new THREE.SpotLight(0xffe3b8, 2.2, 22, Math.PI / 5.2, 0.55, 1.2);
  lamp.position.set(0, 6.2, 0); lamp.target.position.set(0, 0, 0); lamp.castShadow = true;
  lamp.shadow.mapSize.set(2048, 2048); lamp.shadow.bias = -0.0005; lamp.shadow.radius = 4;
  scene.add(lamp, lamp.target);
  const rimBlue = new THREE.PointLight(0x4c8cff, 0.5, 18, 2); rimBlue.position.set(-6, 2.5, 3); scene.add(rimBlue);
  const rimAmber = new THREE.PointLight(0xff9a3c, 0.4, 18, 2); rimAmber.position.set(6, 2.5, 3); scene.add(rimAmber);
  const dealerGlow = new THREE.PointLight(0xffd39a, 0.35, 6, 2); dealerGlow.position.set(0, 1.4, -2.6); scene.add(dealerGlow);

  // ---------- room ------------------------------------------------------
  const carpetTex = srgb(canvasTex(512, 512, (x, w, h) => {
    x.fillStyle = "#150f1c"; x.fillRect(0, 0, w, h);
    x.strokeStyle = "rgba(150,110,60,.13)"; x.lineWidth = 2;
    for (let i = 0; i < 8; i++) for (let j = 0; j < 8; j++) {
      x.beginPath(); x.arc(i * 64 + 32, j * 64 + 32, 22, 0, Math.PI * 2); x.stroke();
      x.beginPath(); x.arc(i * 64 + 32, j * 64 + 32, 8, 0, Math.PI * 2); x.stroke();
    }
    for (let i = 0; i < 3000; i++) { x.fillStyle = `rgba(255,255,255,${Math.random() * 0.03})`; x.fillRect(Math.random() * w, Math.random() * h, 2, 2); }
  }));
  carpetTex.wrapS = carpetTex.wrapT = THREE.RepeatWrapping; carpetTex.repeat.set(10, 10);
  const floor = new THREE.Mesh(new THREE.CircleGeometry(16, 64),
    new THREE.MeshStandardMaterial({ map: carpetTex, roughness: 1 }));
  floor.rotation.x = -Math.PI / 2; floor.position.y = -2.4; floor.receiveShadow = true; scene.add(floor);

  // hanging lamp
  const lampGroup = new THREE.Group();
  const shade = new THREE.Mesh(new THREE.ConeGeometry(0.9, 0.55, 40, 1, true),
    new THREE.MeshStandardMaterial({ color: 0x1b3a2a, roughness: 0.5, metalness: 0.3, side: THREE.DoubleSide }));
  shade.position.y = 6.1; lampGroup.add(shade);
  const shadeInner = new THREE.Mesh(new THREE.ConeGeometry(0.86, 0.5, 40, 1, true),
    new THREE.MeshBasicMaterial({ color: 0xffe9c4, side: THREE.BackSide }));
  shadeInner.position.y = 6.08; lampGroup.add(shadeInner);
  const bulb = new THREE.Mesh(new THREE.SphereGeometry(0.1, 12, 12), new THREE.MeshBasicMaterial({ color: 0xfff3d6 }));
  bulb.position.y = 5.95; lampGroup.add(bulb);
  const cord = new THREE.Mesh(new THREE.CylinderGeometry(0.012, 0.012, 3, 6), new THREE.MeshStandardMaterial({ color: 0x111111 }));
  cord.position.y = 7.8; lampGroup.add(cord);
  const beam = new THREE.Mesh(new THREE.ConeGeometry(3.6, 6.2, 48, 1, true),
    new THREE.MeshBasicMaterial({ color: 0xffe3b8, transparent: true, opacity: 0.045, side: THREE.DoubleSide,
      depthWrite: false, blending: THREE.AdditiveBlending }));
  beam.position.y = 3.0; lampGroup.add(beam);
  scene.add(lampGroup);

  // ---------- table -----------------------------------------------------
  const feltTex = srgb(canvasTex(1024, 1024, (x, w, h) => {
    x.fillStyle = "#1c6b3d"; x.fillRect(0, 0, w, h);
    for (let i = 0; i < 40000; i++) {
      x.fillStyle = `rgba(${Math.random() < .5 ? 0 : 255},${Math.random() < .5 ? 0 : 255},0,${Math.random() * 0.045})`;
      x.fillRect(Math.random() * w, Math.random() * h, 1.5, 1.5);
    }
  }));
  const felt = new THREE.Mesh(new THREE.CylinderGeometry(3.3, 3.3, 0.28, 96),
    new THREE.MeshStandardMaterial({ map: feltTex, roughness: 0.97 }));
  felt.position.y = -0.14; felt.receiveShadow = true; scene.add(felt);

  // lettering and betting circles live on a flat overlay, where a plane's
  // texture maps plainly onto world x/z (canvas up = away from the players)
  const overlayTex = srgb(canvasTex(1024, 1024, (x, w, h) => {
    x.clearRect(0, 0, w, h);
    x.save(); x.translate(w / 2, h / 2);
    x.fillStyle = "rgba(226,197,107,.8)"; x.font = "600 30px Heebo, sans-serif"; x.textAlign = "center"; x.textBaseline = "middle";
    const arcText = (text, r, centre) => {
      const widths = [...text].map(ch => x.measureText(ch).width + 2);
      const total = widths.reduce((a, b) => a + b, 0) / r;
      let a = centre - total / 2;
      [...text].forEach((ch, i) => {
        const half = widths[i] / 2 / r;
        x.save(); x.rotate(a + half); x.translate(0, -r); x.fillText(ch, 0, 0); x.restore();
        a += widths[i] / r;
      });
    };
    arcText("FLY BRAIN BLACKJACK", 268, -Math.PI * 0.2);
    arcText("PAYS 3 TO 2", 268, Math.PI * 0.2);
    x.font = "300 22px Heebo, sans-serif"; x.fillStyle = "rgba(226,197,107,.55)";
    arcText("DEALER STANDS ON 17", 300, 0);
    for (const deg of SEAT_ANGLES) {
      const a = THREE.MathUtils.degToRad(deg);
      const cx = Math.sin(a) * (CHIP_RADIUS / 3.3) * (w / 2), cy = Math.cos(a) * (CHIP_RADIUS / 3.3) * (h / 2);
      x.strokeStyle = "rgba(226,197,107,.5)"; x.lineWidth = 3;
      x.beginPath(); x.arc(cx, cy, 40, 0, Math.PI * 2); x.stroke();
    }
    x.restore();
  }));
  const overlay = new THREE.Mesh(new THREE.CircleGeometry(3.3, 96),
    new THREE.MeshStandardMaterial({ map: overlayTex, transparent: true, roughness: 0.9, depthWrite: false }));
  overlay.rotation.x = -Math.PI / 2; overlay.position.y = 0.003; overlay.receiveShadow = true; scene.add(overlay);
  const rim = new THREE.Mesh(new THREE.TorusGeometry(3.34, 0.16, 20, 120),
    new THREE.MeshStandardMaterial({ color: 0x3d2414, roughness: 0.45, metalness: 0.12 }));
  rim.rotation.x = Math.PI / 2; rim.position.y = 0.03; rim.castShadow = true; rim.receiveShadow = true; scene.add(rim);
  const rimTrim = new THREE.Mesh(new THREE.TorusGeometry(3.34, 0.03, 12, 120),
    new THREE.MeshStandardMaterial({ color: 0xe2c56b, roughness: 0.3, metalness: 0.8 }));
  rimTrim.rotation.x = Math.PI / 2; rimTrim.position.y = 0.19; scene.add(rimTrim);
  const skirt = new THREE.Mesh(new THREE.CylinderGeometry(3.3, 3.1, 0.6, 96, 1, true),
    new THREE.MeshStandardMaterial({ color: 0x241309, roughness: 0.7, side: THREE.DoubleSide }));
  skirt.position.y = -0.58; scene.add(skirt);
  const pedestal = new THREE.Mesh(new THREE.CylinderGeometry(0.55, 1.1, 1.8, 40),
    new THREE.MeshStandardMaterial({ color: 0x1a0f08, roughness: 0.7, metalness: 0.1 }));
  pedestal.position.y = -1.6; pedestal.castShadow = true; scene.add(pedestal);

  // card shoe by the dealer's right, discard tray on the left
  const shoePos = new THREE.Vector3(-1.55, 0.02, -2.15);
  const shoe = new THREE.Group(); shoe.position.copy(shoePos); shoe.rotation.y = 0.35;
  const shoeBody = new THREE.Mesh(new THREE.BoxGeometry(0.62, 0.22, 0.85),
    new THREE.MeshPhysicalMaterial({ color: 0x1a1a1f, roughness: 0.25, metalness: 0.2, clearcoat: 0.6 }));
  shoeBody.position.y = 0.11; shoeBody.castShadow = true; shoe.add(shoeBody);
  const shoeCards = new THREE.Mesh(new THREE.BoxGeometry(0.52, 0.16, 0.74),
    new THREE.MeshStandardMaterial({ color: 0xf1ede2, roughness: 0.8 }));
  shoeCards.position.set(0, 0.13, 0.02); shoe.add(shoeCards);
  scene.add(shoe);
  const discardPos = new THREE.Vector3(1.6, 0.02, -2.15);

  // dealer chip tray
  const tray = new THREE.Group(); tray.position.set(0, 0.0, -1.55);
  const trayBody = new THREE.Mesh(new THREE.BoxGeometry(1.25, 0.09, 0.32),
    new THREE.MeshPhysicalMaterial({ color: 0x111116, roughness: 0.3, clearcoat: 0.7 }));
  trayBody.position.y = 0.045; tray.add(trayBody); scene.add(tray);
  const trayChipsPos = new THREE.Vector3(0, 0.1, -1.55);

  // ---------- chips -----------------------------------------------------
  const CHIP_COLORS = [0xd8d8d8, 0xd42b2b, 0x2b7bd4, 0x2fa34d, 0x111111];
  const chipGeo = new THREE.CylinderGeometry(0.115, 0.115, 0.028, 28);
  const chipMats = CHIP_COLORS.map(c => new THREE.MeshPhysicalMaterial({ color: c, roughness: 0.35, clearcoat: 0.8, clearcoatRoughness: 0.2 }));
  const stripeGeo = new THREE.TorusGeometry(0.115, 0.008, 6, 40);
  const stripeMat = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.5 });
  function makeChip(i) {
    const m = new THREE.Mesh(chipGeo, chipMats[i % chipMats.length]); m.castShadow = true;
    const stripe = new THREE.Mesh(stripeGeo, stripeMat); stripe.rotation.x = Math.PI / 2; m.add(stripe);
    return m;
  }
  function buildStack(group, n, colorSeed) {
    while (group.children.length) group.remove(group.children[0]);
    n = Math.max(0, Math.round(n));
    for (let i = 0; i < n; i++) {
      const col = Math.floor(i / 4);
      const chip = makeChip((colorSeed + col) % CHIP_COLORS.length);
      chip.position.set((col % 2) * 0.26 - 0.13 * Math.min(1, Math.floor(n / 4)), 0.014 + (i % 4) * 0.03, Math.floor(col / 2) * -0.26);
      chip.rotation.y = Math.random() * 0.4;
      group.add(chip);
    }
  }
  const trayStack = new THREE.Group(); trayStack.position.copy(trayChipsPos); scene.add(trayStack);
  buildStack(trayStack, 24, 2);
  function flyChips(from, to, n, delayStep = 90) {
    for (let i = 0; i < n; i++) {
      const chip = makeChip((i + 1) % CHIP_COLORS.length); chip.position.copy(from); scene.add(chip);
      const a = from.clone(), b = to.clone();
      tween(650, (k) => {
        chip.position.lerpVectors(a, b, k); chip.position.y += Math.sin(k * Math.PI) * 0.9;
        chip.rotation.x = k * Math.PI * 3;
      }, { delay: i * delayStep, ease: easeInOut, done: () => scene.remove(chip) });
    }
  }

  // ---------- cards -----------------------------------------------------
  const faceCache = new Map();
  function roundedRect(x, w, h, r) { x.beginPath(); x.moveTo(r, 0); x.arcTo(w, 0, w, h, r); x.arcTo(w, h, 0, h, r); x.arcTo(0, h, 0, 0, r); x.arcTo(0, 0, w, 0, r); x.closePath(); }
  function faceTexture(code) {
    if (faceCache.has(code)) return faceCache.get(code);
    const t = srgb(canvasTex(256, 360, (x, w, h) => {
      roundedRect(x, w, h, 24); x.fillStyle = "#f9f6ee"; x.fill();
      x.strokeStyle = "rgba(0,0,0,.08)"; x.lineWidth = 6; roundedRect(x, w, h, 24); x.stroke();
      const rank = code[0] === "T" ? "10" : code[0];
      const suit = { s: "♠", h: "♥", d: "♦", c: "♣" }[code[1]];
      const red = code[1] === "h" || code[1] === "d";
      x.fillStyle = red ? "#c8102e" : "#181b23";
      const corner = (rot) => {
        x.save(); if (rot) { x.translate(w, h); x.rotate(Math.PI); }
        x.font = "800 62px Heebo, system-ui, sans-serif"; x.textBaseline = "top"; x.textAlign = "left";
        x.fillText(rank, 20, 12); x.font = "54px system-ui, sans-serif"; x.fillText(suit, 22, 76); x.restore();
      };
      corner(false); corner(true);
      x.font = "150px system-ui, sans-serif"; x.textAlign = "center"; x.textBaseline = "middle";
      x.fillText(suit, w / 2, h / 2 + 6);
    }));
    faceCache.set(code, t); return t;
  }
  const backTexture = srgb(canvasTex(256, 360, (x, w, h) => {
    roundedRect(x, w, h, 24); x.fillStyle = "#f9f6ee"; x.fill();
    x.save(); roundedRect(x, w, h, 24); x.clip();
    x.fillStyle = "#1b2b5c"; x.fillRect(14, 14, w - 28, h - 28);
    x.strokeStyle = "rgba(226,197,107,.55)"; x.lineWidth = 2;
    for (let i = -400; i < 400; i += 18) { x.beginPath(); x.moveTo(i, 0); x.lineTo(i + 360, 360); x.stroke(); x.beginPath(); x.moveTo(i + 360, 0); x.lineTo(i, 360); x.stroke(); }
    x.fillStyle = "#1b2b5c"; x.beginPath(); x.arc(w / 2, h / 2, 46, 0, Math.PI * 2); x.fill();
    x.strokeStyle = "#e2c56b"; x.lineWidth = 3; x.beginPath(); x.arc(w / 2, h / 2, 46, 0, Math.PI * 2); x.stroke();
    x.fillStyle = "#e2c56b"; x.font = "800 34px Heebo, sans-serif"; x.textAlign = "center"; x.textBaseline = "middle"; x.fillText("FB", w / 2, h / 2 + 2);
    x.restore();
  }));
  const cardGeo = new THREE.PlaneGeometry(0.5, 0.7);
  const cardEdgeGeo = new THREE.BoxGeometry(0.5, 0.7, 0.012);
  function makeCard(code) {
    const g = new THREE.Group();
    const flip = new THREE.Group(); g.add(flip);
    const body = new THREE.Mesh(cardEdgeGeo, new THREE.MeshStandardMaterial({ color: 0xf3efe4, roughness: 0.9 }));
    body.castShadow = true; flip.add(body);
    const face = new THREE.Mesh(cardGeo, new THREE.MeshStandardMaterial({ map: faceTexture(code === "??" ? "As" : code), roughness: 0.6 }));
    face.position.z = 0.0065; flip.add(face);
    const back = new THREE.Mesh(cardGeo, new THREE.MeshStandardMaterial({ map: backTexture, roughness: 0.6 }));
    back.position.z = -0.0065; back.rotation.y = Math.PI; flip.add(back);
    g.rotation.x = -Math.PI / 2;
    flip.rotation.y = code === "??" ? Math.PI : 0;
    g.userData = { code, flip, face };
    return g;
  }

  class Hand {
    constructor(group) { this.group = group; this.cards = []; this.codes = []; }
    slot(i, n) { return new THREE.Vector3((i - (n - 1) / 2) * 0.32, 0.004 * i, 0); }
    layout() {
      const n = this.cards.length;
      this.cards.forEach((c, i) => {
        const to = this.slot(i, n), rz = (i - (n - 1) / 2) * -0.05;
        const from = c.position.clone(), rz0 = c.rotation.z;
        if (c.userData.arriving) return;
        tween(260, k => { c.position.lerpVectors(from, to, k); c.rotation.z = rz0 + (rz - rz0) * k; });
      });
    }
    update(codes) {
      const joined = codes.join(",");
      if (joined === this.codes.join(",")) return;
      // reveal: the dealer's "??" became a real card
      if (this.codes.length === codes.length && this.codes[1] === "??" && codes[1] !== "??") {
        const card = this.cards[1];
        card.userData.face.material.map = faceTexture(codes[1]); card.userData.face.material.needsUpdate = true;
        const flip = card.userData.flip;
        tween(520, k => { flip.rotation.y = Math.PI * (1 - k); card.position.y = 0.004 + Math.sin(k * Math.PI) * 0.35; }, { ease: easeInOut });
        this.codes = codes.slice(); return;
      }
      if (codes.length < this.codes.length || codes.length === 0) {
        for (const c of this.cards) {
          const from = c.position.clone(); const local = this.group.worldToLocal(discardPos.clone());
          tween(420, k => { c.position.lerpVectors(from, local, k); c.position.y += Math.sin(k * Math.PI) * 0.6; c.rotation.z += 0.08; },
            { ease: easeInOut, done: () => this.group.remove(c) });
        }
        this.cards = []; this.codes = [];
        if (codes.length === 0) return;
      }
      const firstNew = this.cards.length;
      for (let i = firstNew; i < codes.length; i++) {
        const card = makeCard(codes[i]);
        const local = this.group.worldToLocal(shoePos.clone()); local.y += 0.15;
        card.position.copy(local); card.rotation.z = 0.6;
        card.userData.flip.rotation.y = Math.PI;         // leaves the shoe face down
        card.userData.arriving = true;
        this.group.add(card); this.cards.push(card);
        const to = this.slot(i, codes.length), rz = (i - (codes.length - 1) / 2) * -0.05;
        const from = card.position.clone(), faceUp = codes[i] !== "??";
        tween(560, k => {
          card.position.lerpVectors(from, to, k); card.position.y = Math.max(0.02, card.position.y + Math.sin(k * Math.PI) * 0.6);
          card.rotation.z = 0.6 + (rz - 0.6) * k;
          if (faceUp) card.userData.flip.rotation.y = Math.PI * (1 - Math.min(1, k * 1.25));
        }, { ease: easeInOut, delay: (i - firstNew) * 120, done: () => { card.userData.arriving = false; } });
      }
      this.codes = codes.slice();
      this.layout();
    }
  }

  // ---------- the fly ---------------------------------------------------
  const wingTex = canvasTex(256, 128, (x, w, h) => {
    x.clearRect(0, 0, w, h);
    x.fillStyle = "rgba(220,232,255,0.55)";
    x.beginPath(); x.moveTo(0, h * 0.5); x.quadraticCurveTo(w * 0.35, 0, w * 0.98, h * 0.42);
    x.quadraticCurveTo(w * 0.9, h * 0.95, w * 0.3, h * 0.9); x.quadraticCurveTo(w * 0.05, h * 0.85, 0, h * 0.5); x.closePath(); x.fill();
    x.strokeStyle = "rgba(60,70,110,0.55)"; x.lineWidth = 1.6;
    for (const [c1, c2] of [[0.15, 0.25], [0.3, 0.42], [0.45, 0.58], [0.55, 0.74]]) {
      x.beginPath(); x.moveTo(6, h * 0.5); x.quadraticCurveTo(w * 0.5, h * c1, w * 0.95, h * c2); x.stroke();
    }
    x.beginPath(); x.moveTo(w * 0.4, h * 0.3); x.lineTo(w * 0.42, h * 0.72); x.stroke();
    x.beginPath(); x.moveTo(w * 0.65, h * 0.33); x.lineTo(w * 0.68, h * 0.78); x.stroke();
  });
  const wingShape = (() => {
    const s = new THREE.Shape();
    s.moveTo(0, 0); s.quadraticCurveTo(0.24, 0.16, 0.66, 0.05); s.quadraticCurveTo(0.6, -0.16, 0.2, -0.13); s.quadraticCurveTo(0.03, -0.11, 0, 0);
    return s;
  })();
  const wingGeo = new THREE.ShapeGeometry(wingShape, 12);
  {
    const pos = wingGeo.attributes.position, uv = wingGeo.attributes.uv;
    for (let i = 0; i < pos.count; i++) uv.setXY(i, pos.getX(i) / 0.66, (pos.getY(i) + 0.16) / 0.32);
  }
  const bubbleTex = canvasTex(128, 128, (x, w, h) => {
    x.clearRect(0, 0, w, h);
    x.fillStyle = "rgba(255,255,255,0.95)";
    x.beginPath(); x.arc(64, 50, 40, 0, Math.PI * 2); x.fill();
    x.beginPath(); x.arc(40, 96, 10, 0, Math.PI * 2); x.fill();
    x.beginPath(); x.arc(28, 116, 5, 0, Math.PI * 2); x.fill();
    x.fillStyle = "#5adcff";
    for (let i = 0; i < 3; i++) { x.beginPath(); x.arc(44 + i * 20, 50, 6, 0, Math.PI * 2); x.fill(); }
  });

  function makeFly({ scale = 1, dealer = false, golden = false } = {}) {
    const g = new THREE.Group();
    const bodyColor = golden ? 0xd4a83a : dealer ? 0x2a2622 : 0xa86a36;
    const abdColor = golden ? 0xb88b2c : dealer ? 0x1a1714 : 0x7a4a24;
    const bodyMat = new THREE.MeshPhysicalMaterial({ color: bodyColor, roughness: 0.55, metalness: golden ? 0.5 : 0.05, clearcoat: 0.35, clearcoatRoughness: 0.5 });
    const abdMat = new THREE.MeshPhysicalMaterial({ color: abdColor, roughness: 0.6, metalness: golden ? 0.45 : 0.05, clearcoat: 0.3 });
    const darkMat = new THREE.MeshStandardMaterial({ color: 0x1a120c, roughness: 0.75 });

    const thorax = new THREE.Mesh(new THREE.SphereGeometry(0.22, 28, 20), bodyMat);
    thorax.scale.set(1, 0.92, 1.15); thorax.castShadow = true; g.add(thorax);
    for (let i = 0; i < 10; i++) {
      const b = new THREE.Mesh(new THREE.CylinderGeometry(0.003, 0.007, 0.09, 4), darkMat);
      const a = (i / 10) * Math.PI * 2;
      b.position.set(Math.cos(a) * 0.14, 0.19, Math.sin(a) * 0.12); b.rotation.set(Math.sin(a) * 0.5, 0, -Math.cos(a) * 0.5); g.add(b);
    }
    const abdomen = new THREE.Mesh(new THREE.SphereGeometry(0.25, 28, 20), abdMat);
    abdomen.scale.set(0.92, 0.8, 1.6); abdomen.position.set(0, -0.03, -0.44); abdomen.castShadow = true; g.add(abdomen);
    for (let i = 0; i < 4; i++) {
      const stripe = new THREE.Mesh(new THREE.TorusGeometry(0.215 - i * 0.03, 0.02, 8, 36), darkMat);
      stripe.position.set(0, -0.03, -0.3 - i * 0.12); stripe.scale.set(0.95, 0.75, 1); g.add(stripe);
    }

    const head = new THREE.Group(); head.position.set(0, 0.05, 0.31); g.add(head);
    const skull = new THREE.Mesh(new THREE.SphereGeometry(0.165, 28, 20), bodyMat); skull.scale.set(1.05, 1, 0.95); skull.castShadow = true; head.add(skull);
    const eyeMat = new THREE.MeshStandardMaterial({ color: 0xff2020, emissive: 0xff1010, emissiveIntensity: golden ? 0.6 : 0.85, roughness: 0.25 });
    const eyes = [];
    for (const s of [-1, 1]) {
      const eye = new THREE.Mesh(new THREE.SphereGeometry(0.1, 24, 18), eyeMat);
      eye.position.set(s * 0.115, 0.03, 0.06); eye.scale.set(0.95, 1.1, 1); head.add(eye); eyes.push(eye);
      const glint = new THREE.Mesh(new THREE.SphereGeometry(0.022, 8, 8), new THREE.MeshBasicMaterial({ color: 0xffffff }));
      glint.position.set(s * 0.15, 0.075, 0.13); head.add(glint);
      const antenna = new THREE.Group(); antenna.position.set(s * 0.05, 0.13, 0.11); head.add(antenna);
      const seg = new THREE.Mesh(new THREE.CylinderGeometry(0.008, 0.014, 0.12, 6), darkMat); seg.position.y = 0.06; antenna.add(seg);
      const arista = new THREE.Mesh(new THREE.CylinderGeometry(0.002, 0.004, 0.12, 4), darkMat); arista.position.set(s * 0.03, 0.14, 0); arista.rotation.z = -s * 0.7; antenna.add(arista);
      antenna.rotation.set(-0.5, 0, -s * 0.35); head.userData[s < 0 ? "antL" : "antR"] = antenna;
    }
    const proboscis = new THREE.Mesh(new THREE.CylinderGeometry(0.025, 0.045, 0.12, 10), darkMat);
    proboscis.position.set(0, -0.12, 0.06); proboscis.rotation.x = 0.35; head.add(proboscis);

    const legs = [];
    for (let i = 0; i < 3; i++) for (const s of [-1, 1]) {
      const hip = new THREE.Group(); hip.position.set(s * 0.15, -0.09, 0.13 - i * 0.15);
      const femur = new THREE.Mesh(new THREE.CylinderGeometry(0.014, 0.018, 0.28, 7), darkMat);
      femur.position.set(s * 0.12, -0.04, 0); femur.rotation.z = s * 1.15; hip.add(femur);
      const knee = new THREE.Group(); knee.position.set(s * 0.24, -0.09, 0); hip.add(knee);
      const tibia = new THREE.Mesh(new THREE.CylinderGeometry(0.008, 0.013, 0.3, 6), darkMat);
      tibia.position.set(s * 0.02, -0.15, 0); tibia.rotation.z = s * 0.15; knee.add(tibia);
      const tarsus = new THREE.Mesh(new THREE.CylinderGeometry(0.005, 0.008, 0.1, 5), darkMat);
      tarsus.position.set(s * 0.05, -0.32, 0.02); tarsus.rotation.z = s * 0.5; knee.add(tarsus);
      g.add(hip); legs.push({ hip, knee, side: s, pair: i });
    }

    const wingMat = new THREE.MeshPhysicalMaterial({ map: wingTex, transparent: true, opacity: 0.85, roughness: 0.1, metalness: 0.15,
      side: THREE.DoubleSide, clearcoat: 1, clearcoatRoughness: 0.05, depthWrite: false, color: 0xe8f0ff });
    const wings = [];
    for (const s of [-1, 1]) {
      const pivot = new THREE.Group(); pivot.position.set(s * 0.09, 0.2, -0.05);
      const w = new THREE.Mesh(wingGeo, wingMat);
      w.rotation.x = -Math.PI / 2; w.scale.set(s, 1, 1); w.rotation.z = s * 0.12; pivot.add(w);
      pivot.rotation.z = s * 0.2; g.add(pivot); wings.push({ pivot, side: s });
      const haltere = new THREE.Group(); haltere.position.set(s * 0.17, 0.02, -0.2);
      const stalk = new THREE.Mesh(new THREE.CylinderGeometry(0.004, 0.004, 0.1, 4), darkMat); stalk.rotation.z = s * 1.2; stalk.position.x = s * 0.04; haltere.add(stalk);
      const knob = new THREE.Mesh(new THREE.SphereGeometry(0.02, 8, 8), new THREE.MeshStandardMaterial({ color: 0xd9c46a })); knob.position.set(s * 0.085, -0.02, 0); haltere.add(knob);
      g.add(haltere);
    }

    if (dealer) {
      const tieMat = new THREE.MeshPhysicalMaterial({ color: 0xc41e3a, roughness: 0.35, clearcoat: 0.6 });
      for (const s of [-1, 1]) {
        const half = new THREE.Mesh(new THREE.ConeGeometry(0.065, 0.13, 4), tieMat);
        half.position.set(s * 0.08, -0.09, 0.25); half.rotation.z = s * Math.PI / 2; half.rotation.y = Math.PI / 4; g.add(half);
      }
      const knot = new THREE.Mesh(new THREE.SphereGeometry(0.032, 10, 10), tieMat); knot.position.set(0, -0.09, 0.26); g.add(knot);
      const visor = new THREE.Mesh(new THREE.CylinderGeometry(0.21, 0.21, 0.05, 28, 1, false, 0, Math.PI),
        new THREE.MeshPhysicalMaterial({ color: 0x1f6b3a, transparent: true, opacity: 0.85, clearcoat: 1 }));
      visor.position.set(0, 0.17, 0.09); visor.rotation.set(0.35, Math.PI, 0); head.add(visor);
      const band = new THREE.Mesh(new THREE.TorusGeometry(0.17, 0.012, 8, 40), new THREE.MeshStandardMaterial({ color: 0x111111 }));
      band.rotation.x = Math.PI / 2; band.position.y = 0.16; head.add(band);
    }

    const bubble = new THREE.Sprite(new THREE.SpriteMaterial({ map: bubbleTex, transparent: true, depthWrite: false }));
    bubble.position.set(0.28, 0.62, 0.15); bubble.scale.setScalar(0); g.add(bubble);

    g.scale.setScalar(scale);
    return { group: g, head, eyes, legs, wings, bubble, phase: Math.random() * 10, anim: "idle", animStart: 0,
      baseY: 0, baseRotY: 0, nextBlink: 2 + Math.random() * 3, blinkUntil: 0, nextGroom: 4 + Math.random() * 8, groomUntil: 0 };
  }

  // ---------- particles -------------------------------------------------
  function sparkle(at, color = 0x6ff0a8, n = 42) {
    const geo = new THREE.BufferGeometry();
    const pos = new Float32Array(n * 3), vel = [];
    for (let i = 0; i < n; i++) {
      pos.set([at.x, at.y, at.z], i * 3);
      const a = Math.random() * Math.PI * 2, r = 0.6 + Math.random() * 1.2;
      vel.push(new THREE.Vector3(Math.cos(a) * r, 1.6 + Math.random() * 1.6, Math.sin(a) * r));
    }
    geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    const mat = new THREE.PointsMaterial({ color, size: 0.07, transparent: true, opacity: 1, depthWrite: false, blending: THREE.AdditiveBlending });
    const pts = new THREE.Points(geo, mat); scene.add(pts);
    tween(1300, (k, raw) => {
      const t = raw * 1.3;
      for (let i = 0; i < n; i++) {
        pos[i * 3] = at.x + vel[i].x * t; pos[i * 3 + 1] = at.y + vel[i].y * t - 2.6 * t * t; pos[i * 3 + 2] = at.z + vel[i].z * t;
      }
      geo.attributes.position.needsUpdate = true; mat.opacity = 1 - raw;
    }, { ease: k => k, done: () => scene.remove(pts) });
  }

  // ---------- seats -----------------------------------------------------
  function makeTurnRing() {
    const m = new THREE.Mesh(new THREE.RingGeometry(0.72, 0.84, 64),
      new THREE.MeshBasicMaterial({ color: 0xffe084, transparent: true, opacity: 0, side: THREE.DoubleSide, depthWrite: false, blending: THREE.AdditiveBlending }));
    m.rotation.x = -Math.PI / 2; m.userData.on = false; return m;
  }
  const seats = SEAT_ANGLES.map((deg, i) => {
    const a = THREE.MathUtils.degToRad(deg);
    const pos = new THREE.Vector3(SEAT_RADIUS * Math.sin(a), 0.24, SEAT_RADIUS * Math.cos(a));
    const cards = new THREE.Group(); cards.position.set(CARD_RADIUS * Math.sin(a), 0.012, CARD_RADIUS * Math.cos(a)); cards.rotation.y = a; scene.add(cards);
    const chips = new THREE.Group(); chips.position.set(CHIP_RADIUS * Math.sin(a), 0.0, CHIP_RADIUS * Math.cos(a)); chips.rotation.y = a; scene.add(chips);
    const label = document.createElement("div"); label.className = "label glass"; document.getElementById("labels").appendChild(label);
    const ring = makeTurnRing(); ring.position.set(CARD_RADIUS * Math.sin(a), 0.006, CARD_RADIUS * Math.cos(a)); scene.add(ring);
    return { index: i, angle: a, pos, hand: new Hand(cards), chips, chipCount: null, label, fly: null, kind: "empty", brainKey: "", lastResultKey: "", ring };
  });
  const dealer = (() => {
    const f = makeFly({ scale: 1.4, dealer: true });
    f.group.position.set(0, 0.3, -2.85); f.baseY = 0.3; f.baseRotY = 0; scene.add(f.group);
    const cards = new THREE.Group(); cards.position.set(0, 0.012, -0.95); cards.rotation.y = Math.PI; scene.add(cards);
    const label = document.createElement("div"); label.className = "label glass"; document.getElementById("labels").appendChild(label);
    const ring = makeTurnRing(); ring.position.set(0, 0.006, -0.95); scene.add(ring);
    return { fly: f, hand: new Hand(cards), label, pos: f.group.position, ring };
  })();
  const turnRings = [...seats.map(s => s.ring), dealer.ring];

  function setFly(seat, kind) {
    if (seat.fly) {
      const old = seat.fly; const from = old.group.position.clone();
      tween(500, k => { old.group.position.y = from.y + k * 1.4; old.group.scale.setScalar((1 - k) * 0.95 + 0.001); }, { done: () => scene.remove(old.group) });
      seat.fly = null;
    }
    if (kind === "empty") return;
    const f = makeFly(kind === "human" ? { golden: true, scale: 0.95 } : { scale: 0.95 });
    f.group.position.copy(seat.pos); f.baseRotY = seat.angle + Math.PI; f.group.rotation.y = f.baseRotY; f.baseY = seat.pos.y;
    f.group.userData.seat = seat.index; f.group.scale.setScalar(0.001);
    scene.add(f.group); seat.fly = f;
    tween(600, k => { f.group.scale.setScalar(0.95 * k); f.group.position.y = seat.pos.y + (1 - k) * 1.2; }, { ease: easeOut });
  }

  // ---------- fly animation ---------------------------------------------
  const clock = new THREE.Clock();
  const bubbleTarget = new THREE.Vector3();
  function animateFly(f, t) {
    const g = f.group, dt = t - f.animStart;
    let y = f.baseY + 0.015 * Math.sin(t * 2.2 + f.phase), rx = 0, ry = 0, flutter = 0, headTilt = 0, headNod = 0;
    let bubble = 0, antTwitch = 0, legWave = 0.03;
    switch (f.anim) {
      case "thinking": flutter = 0.55; rx = 0.16; headTilt = 0.3 * Math.sin(t * 3.2 + f.phase); headNod = 0.1 * Math.sin(t * 7);
        y += 0.05 * Math.sin(t * 6); bubble = 1; antTwitch = 0.5; legWave = 0.15; break;
      case "waiting": flutter = 0.15; headTilt = 0.25 * Math.sin(t * 2); antTwitch = 0.2; break;
      case "hit": y += 0.22 * Math.max(0, Math.sin(Math.min(dt, 0.5) * Math.PI * 2)); rx = 0.3 * Math.exp(-dt * 2.5);
        flutter = 0.45 * Math.exp(-dt * 2) + 0.02; headNod = 0.35 * Math.exp(-dt * 3) * Math.sin(dt * 18); break;
      case "stand": rx = -0.25 * Math.exp(-dt * 1.6); headTilt = 0.2 * Math.exp(-dt * 2); break;
      case "bust": case "lost": rx = 0.5; y -= 0.06; flutter = 0; headNod = 0.4; break;
      case "won": case "blackjack": y += 0.36 * Math.abs(Math.sin(dt * 7)) * Math.exp(-dt * 0.8);
        ry = dt < 1.25 ? dt * Math.PI * 2 * 0.8 : 0; flutter = 0.7 * Math.exp(-dt * 0.7) + 0.05; headNod = -0.2; break;
      case "push": headTilt = 0.35; break;
    }
    if (t > f.nextBlink) { f.blinkUntil = t + 0.12; f.nextBlink = t + 2.5 + Math.random() * 4; }
    const blink = t < f.blinkUntil ? 0.12 : 1;
    for (const e of f.eyes) e.scale.y = 1.1 * blink;
    if (f.anim === "idle" && t > f.nextGroom) { f.groomUntil = t + 2.2; f.nextGroom = t + 7 + Math.random() * 9; }
    const grooming = f.anim === "idle" && t < f.groomUntil;
    if (grooming) { headNod = 0.35; rx = 0.12; }

    g.position.y = y; g.rotation.x = rx; g.rotation.y = f.baseRotY + ry;
    f.head.rotation.z = headTilt; f.head.rotation.x = headNod;
    const antL = f.head.userData.antL, antR = f.head.userData.antR;
    antL.rotation.x = -0.5 + antTwitch * 0.25 * Math.sin(t * 11 + f.phase); antR.rotation.x = -0.5 + antTwitch * 0.25 * Math.sin(t * 13);
    for (const w of f.wings) {
      const beat = flutter > 0.03 ? Math.sin(t * 46 + f.phase) * flutter : 0.04 * Math.sin(t * 2.5 + f.phase);
      w.pivot.rotation.z = w.side * (0.2 + beat); w.pivot.rotation.y = w.side * -0.2 * flutter;
      if (f.anim === "bust" || f.anim === "lost") w.pivot.rotation.z = w.side * -0.3;
    }
    for (const L of f.legs) {
      const base = L.pair === 0 ? 0.35 : L.pair === 1 ? 0 : -0.35;
      if (grooming && L.pair === 0) {
        L.hip.rotation.x = base + 0.4 + 0.25 * Math.sin(t * 14 + (L.side < 0 ? 0 : Math.PI)); L.hip.rotation.z = L.side * -0.55; L.knee.rotation.z = L.side * 0.9;
      } else {
        L.hip.rotation.x = base + legWave * Math.sin(t * 9 + L.pair * 1.3 + (L.side < 0 ? 0 : 1)); L.hip.rotation.z = 0; L.knee.rotation.z = 0;
      }
    }
    const bs = bubble ? 0.55 + 0.05 * Math.sin(t * 5) : 0;
    f.bubble.scale.lerp(bubbleTarget.set(bs, bs, bs), 0.15);
  }
  function setAnim(f, anim, t, seat) {
    if (f.anim === anim) return;
    f.anim = anim; f.animStart = t;
    if ((anim === "won" || anim === "blackjack") && seat) sparkle(seat.pos.clone().add(new THREE.Vector3(0, 0.3, 0)), anim === "blackjack" ? 0xe2c56b : 0x6ff0a8);
  }

  // ---------- brain panel -----------------------------------------------
  const brainBox = document.getElementById("brain");
  const kcCanvas = document.getElementById("kc"), kcCtx = kcCanvas.getContext("2d");
  const voteCanvas = document.getElementById("votes"), voteCtx = voteCanvas.getContext("2d");
  const COLS = 98, CELL = 3, N_KC = 5177;
  let brain = null, pinnedSeat = null;

  function startBrain(seat, data) {
    brain = { frames: data.kc_frames, votes: data.votes, action: data.action, who: seat.label.dataset.name,
      total: data.total, dealer: data.dealer, start: performance.now(), glow: new Float32Array(N_KC), ever: new Uint8Array(N_KC), drawn: -1 };
    brainBox.classList.remove("hidden");
    document.getElementById("brain-who").innerHTML = `המוח של ${brain.who}` + (pinnedSeat !== null ? ` <span class="pin">· מוצמד</span>` : "");
    document.getElementById("brain-what").textContent = `${brain.total} מול ${brain.dealer === 11 ? "A" : brain.dealer}`;
  }
  function drawBrain() {
    if (!brain) return;
    const n = brain.frames.length;
    const idx = Math.min(n - 1, Math.floor((performance.now() - brain.start) / BRAIN_PLAY_MS * n));
    if (idx === brain.drawn) return;
    for (let f = brain.drawn + 1; f <= idx; f++) {
      for (let i = 0; i < N_KC; i++) brain.glow[i] *= 0.8;
      for (const k of brain.frames[f]) { brain.glow[k] = 1; brain.ever[k] = 1; }
    }
    brain.drawn = idx;
    const W = kcCanvas.width, H = kcCanvas.height;
    kcCtx.fillStyle = "#070a11"; kcCtx.fillRect(0, 0, W, H);
    const ox = (W - COLS * CELL) / 2, oy = (H - Math.ceil(N_KC / COLS) * CELL) / 2;
    let ever = 0;
    for (let i = 0; i < N_KC; i++) {
      const g = brain.glow[i], x = ox + (i % COLS) * CELL, y = oy + Math.floor(i / COLS) * CELL;
      if (g > 0.03) { kcCtx.fillStyle = `rgba(90,220,255,${0.3 + 0.7 * g})`; kcCtx.fillRect(x, y, CELL - 0.6, CELL - 0.6); }
      else if (brain.ever[i]) { kcCtx.fillStyle = "rgba(90,180,255,0.16)"; kcCtx.fillRect(x, y, CELL - 0.6, CELL - 0.6); }
      if (brain.ever[i]) ever++;
    }
    kcCtx.globalCompositeOperation = "lighter"; kcCtx.filter = "blur(3px)"; kcCtx.globalAlpha = 0.35;
    kcCtx.drawImage(kcCanvas, 0, 0); kcCtx.filter = "none"; kcCtx.globalAlpha = 1; kcCtx.globalCompositeOperation = "source-over";
    document.getElementById("kc-count").textContent = ever.toLocaleString();
    document.getElementById("kc-step").textContent = idx + 1;

    const v = brain.votes[idx], VW = voteCanvas.width, mid = VW / 2, scale = 34, BH = 18;
    voteCtx.clearRect(0, 0, VW, voteCanvas.height);
    voteCtx.fillStyle = "rgba(255,255,255,.12)"; voteCtx.fillRect(mid - 0.5, 4, 1, 50);
    const rows = [["HIT", "#ff9d47", 6], ["STAND", "#63b8ff", 32]];
    for (const [name, color, y] of rows) {
      const val = Math.max(-3.8, Math.min(3.8, v[name] || 0)), len = val * scale;
      const winner = idx === n - 1 && name === brain.action;
      voteCtx.shadowBlur = winner ? 14 : 0; voteCtx.shadowColor = color;
      voteCtx.fillStyle = winner ? color : color + "88";
      if (len >= 0) voteCtx.fillRect(mid, y, len, BH); else voteCtx.fillRect(mid + len, y, -len, BH);
      voteCtx.shadowBlur = 0;
      voteCtx.fillStyle = "#eef0f5"; voteCtx.font = `${winner ? 800 : 500} 12px Heebo, system-ui`; voteCtx.textBaseline = "middle";
      voteCtx.textAlign = "right"; voteCtx.fillText(name, mid - 6 - Math.max(0, -len), y + BH / 2);
      voteCtx.textAlign = "left"; voteCtx.fillText(val.toFixed(2), mid + 6 + Math.max(0, len), y + BH / 2);
    }
  }

  // ---------- state -----------------------------------------------------
  let state = null, joined = false, lastMessage = "";
  // remembered across reloads so a refresh does not orphan the human seat
  let myName = null; try { myName = localStorage.getItem("flyName"); } catch (e) {}
  const STATUS_HE = { idle: "", waiting: "תורך", thinking: "חושב...", hit: "HIT", stand: "STAND",
    bust: "נשרף", won: "ניצח", lost: "הפסיד", push: "תיקו", blackjack: "בלאקג'ק!" };
  const toast = document.getElementById("toast");
  function showToast(text, cls) {
    toast.textContent = text; toast.className = `show ${cls}`;
    setTimeout(() => toast.classList.remove("show"), 1500);
  }

  function applyState(s) {
    state = s;
    const t = clock.getElapsedTime();
    const msgEl = document.getElementById("message");
    if (s.message !== lastMessage) { msgEl.textContent = s.message || ""; msgEl.classList.remove("flash"); void msgEl.offsetWidth; msgEl.classList.add("flash"); lastMessage = s.message; }
    document.getElementById("round").innerHTML = `סיבוב <b>${s.round}</b>` + (s.learning ? ` · <span style="color:var(--win)">לומדים</span>` : "");

    dealer.hand.update(s.dealer.cards);
    dealer.label.innerHTML = `<span class="name">הדילר</span>` + (s.dealer.total != null ? `<span class="total">${s.dealer.total}</span>` : "");
    setAnim(dealer.fly, s.phase === "dealer" ? "thinking" : "idle", t);
    const dealerActive = s.phase === "dealer";
    dealer.ring.userData.on = dealerActive;
    dealer.label.classList.toggle("active", dealerActive);
    // overhead for the dealer's turn and the payout, back to the room for the deal
    if (dealerActive || s.phase === "settle") setView("dealer", DEALER_VIEW);
    else if (s.phase === "dealing" || s.phase === "starting") setView("overview", OVERVIEW);

    s.seats.forEach((ss, i) => {
      const seat = seats[i]; if (!seat) return;
      if (seat.kind !== ss.kind) { seat.kind = ss.kind; setFly(seat, ss.kind); seat.chipCount = null; }
      seat.label.dataset.name = ss.name;
      if (ss.kind === "empty") { seat.label.style.display = "none"; seat.ring.userData.on = false; seat.hand.update([]); if (seat.chipCount !== 0) { buildStack(seat.chips, 0, 0); seat.chipCount = 0; } return; }
      seat.label.style.display = "";
      seat.label.classList.toggle("me", ss.kind === "human" && ss.name === myName);
      const active = ss.status === "thinking" || ss.status === "waiting" || ss.status === "hit";
      seat.label.classList.toggle("active", active);
      seat.ring.userData.on = active;
      if (active) setView(`seat${i}`, seatView(seat.angle));
      const chips = ss.chips, cls = chips > 0 ? "pos" : chips < 0 ? "neg" : "";
      seat.label.innerHTML = `<span class="name">${ss.name}</span><span class="chips ${cls}">${chips > 0 ? "+" : ""}${chips}</span>` +
        `<span class="status ${ss.status}">${STATUS_HE[ss.status] ?? ""}${ss.cards.length ? `<span class="total">${ss.total}</span>` : ""}</span>`;
      seat.hand.update(ss.cards);
      if (seat.fly) setAnim(seat.fly, ss.status === "idle" ? "idle" : ss.status, t, seat);

      const stack = START_STACK + chips;
      const resultKey = `${s.round}:${ss.result}`;
      if (ss.result != null && seat.lastResultKey !== resultKey) {
        seat.lastResultKey = resultKey;
        const n = Math.max(1, Math.round(Math.abs(ss.result)));
        const seatWorld = new THREE.Vector3(CHIP_RADIUS * Math.sin(seat.angle), 0.1, CHIP_RADIUS * Math.cos(seat.angle));
        if (ss.result > 0) flyChips(trayChipsPos, seatWorld, n); else if (ss.result < 0) flyChips(seatWorld, trayChipsPos, n);
        setTimeout(() => { buildStack(seat.chips, stack, i + 1); seat.chipCount = stack; }, 700);
        if (ss.kind === "human" && ss.name === myName) showToast(ss.result > 0 ? (ss.status === "blackjack" ? "בלאקג'ק!" : "ניצחת!") : ss.result < 0 ? "הפסדת" : "תיקו", ss.result > 0 ? "win" : ss.result < 0 ? "lose" : "push");
      } else if (seat.chipCount === null) { buildStack(seat.chips, stack, i + 1); seat.chipCount = stack; }

      if (ss.brain) {
        const key = `${s.round}:${ss.cards.join(",")}:${ss.brain.action}`;
        if (seat.brainKey !== key && (pinnedSeat === null || pinnedSeat === i)) { seat.brainKey = key; startBrain(seat, ss.brain); }
      }
    });

    const humanSeat = s.seats[s.seats.length - 1];
    joined = humanSeat.kind === "human" && humanSeat.name === myName;
    document.getElementById("join-box").style.display = humanSeat.kind === "empty" && !joined ? "" : "none";
    document.getElementById("play-box").style.display = joined ? "" : "none";
    document.getElementById("takeover-box").style.display = humanSeat.kind === "human" && !joined ? "" : "none";
    const myTurn = joined && s.human_turn;
    document.getElementById("hit").disabled = !myTurn; document.getElementById("stand").disabled = !myTurn;
    const statusEl = document.getElementById("status");
    const activeSeat = s.seats.find(x => x.kind !== "empty" && (x.status === "thinking" || x.status === "waiting" || x.status === "hit"));
    const whose = dealerActive ? "הדילר משחק" : activeSeat ? `התור של ${activeSeat.name}` : s.phase === "settle" ? "סיכום היד" : "מחלקים";
    if (myTurn) { statusEl.innerHTML = `<span class="pulse"></span>תורך! יש לך ${humanSeat.total}`; statusEl.className = "turn"; }
    else if (joined) { statusEl.textContent = `אתה בשולחן · ${whose}`; statusEl.className = ""; }
    else if (humanSeat.kind === "human") { statusEl.textContent = `${humanSeat.name} יושב בכיסא · ${whose}`; statusEl.className = ""; }
    else { statusEl.textContent = `אתה צופה · ${whose}`; statusEl.className = ""; }
    document.getElementById("hint").innerHTML = myTurn ? `<kbd>H</kbd> לקלף · <kbd>S</kbd> לעמוד`
      : "גרור לסובב · גלגלת לזום · לחץ על זבוב כדי לראות את המוח שלו";
  }

  async function poll() {
    try { const r = await fetch("/state"); applyState(await r.json()); }
    catch (e) { document.getElementById("message").textContent = "השרת לא זמין"; }
    setTimeout(poll, POLL_MS);
  }

  document.getElementById("join").onclick = async () => {
    myName = (document.getElementById("name").value || "You").trim().slice(0, 16);
    try { localStorage.setItem("flyName", myName); } catch (e) {}
    await fetch("/join", { method: "POST", body: JSON.stringify({ name: myName }) });
  };
  document.getElementById("name").addEventListener("keydown", e => { if (e.key === "Enter" || e.keyCode === 13) document.getElementById("join").click(); });
  document.getElementById("leave").onclick = async () => {
    await fetch("/leave", { method: "POST", body: "{}" }); myName = null;
    try { localStorage.removeItem("flyName"); } catch (e) {}
  };
  // someone else's stale seat (or ours from a browser that forgot): offer to take it over
  document.getElementById("takeover").onclick = async () => {
    await fetch("/leave", { method: "POST", body: "{}" });
    document.getElementById("join").click();
  };
  document.getElementById("hit").onclick = () => fetch("/action", { method: "POST", body: JSON.stringify({ action: "HIT" }) });
  document.getElementById("stand").onclick = () => fetch("/action", { method: "POST", body: JSON.stringify({ action: "STAND" }) });
  addEventListener("keydown", (e) => {
    if (!joined || !state?.human_turn || e.target.tagName === "INPUT") return;
    if (e.key.toLowerCase() === "h") document.getElementById("hit").click();
    if (e.key.toLowerCase() === "s") document.getElementById("stand").click();
  });

  const raycaster = new THREE.Raycaster(), mouse = new THREE.Vector2();
  let downAt = null;
  renderer.domElement.addEventListener("pointerdown", (e) => { downAt = [e.clientX, e.clientY]; });
  renderer.domElement.addEventListener("pointerup", (e) => {
    if (!downAt || Math.hypot(e.clientX - downAt[0], e.clientY - downAt[1]) > 6) return;
    mouse.set((e.clientX / innerWidth) * 2 - 1, -(e.clientY / innerHeight) * 2 + 1);
    raycaster.setFromCamera(mouse, camera);
    const flies = seats.filter(s => s.fly && s.kind === "fly").map(s => s.fly.group);
    const hit = raycaster.intersectObjects(flies, true)[0];
    if (!hit) { pinnedSeat = null; document.querySelector("#brain-who .pin")?.remove(); return; }
    let o = hit.object; while (o && o.userData.seat === undefined) o = o.parent;
    if (!o) return;
    pinnedSeat = o.userData.seat;
    const ss = state?.seats[pinnedSeat];
    if (ss?.brain) { seats[pinnedSeat].brainKey = ""; startBrain(seats[pinnedSeat], ss.brain); }
  });

  // ---------- render loop -----------------------------------------------
  const tmp = new THREE.Vector3();
  const controlsEl = document.getElementById("controls");
  function placeLabel(label, pos, lift) {
    tmp.copy(pos); tmp.y += lift; tmp.project(camera);
    if (tmp.z > 1) { label.style.opacity = "0"; return; }
    const w = label.offsetWidth || 100, h = label.offsetHeight || 40;
    let x = (tmp.x + 1) / 2 * innerWidth, y = (1 - tmp.y) / 2 * innerHeight;
    // stay on screen (the label is anchored at its bottom centre)
    x = Math.min(Math.max(x, w / 2 + 8), innerWidth - w / 2 - 8);
    y = Math.min(Math.max(y, h + 60), innerHeight - 8);
    // stay above the panels rather than under them
    for (const el of [controlsEl, brainBox]) {
      if (el.classList.contains("hidden")) continue;
      const b = el.getBoundingClientRect();
      if (y > b.top - 4 && x + w / 2 > b.left && x - w / 2 < b.right) y = b.top - 4;
    }
    label.style.left = `${x}px`; label.style.top = `${y}px`;
    label.style.opacity = "1";
  }
  let lastFrame = performance.now();
  function render() {
    requestAnimationFrame(render);
    const t = clock.getElapsedTime(), now = performance.now();
    const dt = Math.min(0.1, (now - lastFrame) / 1000); lastFrame = now;
    const idle = now - lastInteraction > 6000;
    if (follow && idle) {
      controls.autoRotate = false;
      // frame-rate independent glide: ~90% of the way in a second
      const k = 1 - Math.exp(-dt * 2.4);
      camera.position.lerp(view.pos, k); controls.target.lerp(view.target, k);
    } else if (!follow && !controls.autoRotate && now - lastInteraction > 20000) controls.autoRotate = true;
    for (const ring of turnRings) {
      const target = ring.userData.on ? 0.55 + 0.3 * Math.sin(t * 5) : 0;
      ring.material.opacity += (target - ring.material.opacity) * 0.12;
      const sc = ring.userData.on ? 1 + 0.04 * Math.sin(t * 5) : 1; ring.scale.set(sc, sc, 1);
    }
    controls.update(); runTweens(now);
    // the overhead view sits above the lamp; do not look through the shade
    const aboveLamp = camera.position.y > 5.0;
    shade.visible = shadeInner.visible = bulb.visible = cord.visible = !aboveLamp;
    beam.material.opacity = aboveLamp ? 0.012 : 0.045;
    for (const s of seats) if (s.fly) animateFly(s.fly, t);
    animateFly(dealer.fly, t);
    bulb.material.color.setHSL(0.1, 0.5, 0.92 + 0.03 * Math.sin(t * 17));
    composer.render();
    for (const s of seats) if (s.kind !== "empty") placeLabel(s.label, s.pos, 0.66);
    placeLabel(dealer.label, dealer.pos, 0.95);
    drawBrain();
  }
  addEventListener("resize", () => {
    camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix();
    renderer.setSize(innerWidth, innerHeight); composer.setSize(innerWidth, innerHeight);
  });

  window.__table = { seats, dealer, scene, camera };
  render(); poll();
})();
