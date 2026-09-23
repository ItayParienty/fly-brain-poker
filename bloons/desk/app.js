/* The fly at its desk - 3D view.
   Streams frames from desk_server.py: the game's screen goes onto the monitor,
   the fly's head follows its gaze (which is the pointer), its proboscis
   extends when it presses, its wings beat when it starts a round, and the
   side panel shows what its eye sees and where it wants to look. */

(() => {
  // ---------- helpers ---------------------------------------------------
  const tweens = [];
  const easeOut = k => 1 - Math.pow(1 - k, 3);
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
  const srgb = tex => { tex.encoding = THREE.sRGBEncoding; return tex; };
  const canvasTex = (w, h, draw) => {
    const c = document.createElement("canvas"); c.width = w; c.height = h;
    draw(c.getContext("2d"), w, h);
    const t = new THREE.CanvasTexture(c); t.anisotropy = 8; return t;
  };
  const fromB64 = s => Uint8Array.from(atob(s), ch => ch.charCodeAt(0));

  // ---------- renderer / scene ------------------------------------------
  const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 1.75));
  renderer.setSize(innerWidth, innerHeight);
  renderer.shadowMap.enabled = true; renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.toneMapping = THREE.ACESFilmicToneMapping; renderer.toneMappingExposure = 1.0;
  document.getElementById("scene").appendChild(renderer.domElement);
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x06070b);
  scene.fog = new THREE.FogExp2(0x06070b, 0.05);

  const camera = new THREE.PerspectiveCamera(42, innerWidth / innerHeight, 0.05, 60);
  const controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true; controls.dampingFactor = 0.07;
  controls.minDistance = 0.8; controls.maxDistance = 9; controls.maxPolarAngle = Math.PI * 0.55;
  let lastInteraction = -1e9;
  for (const ev of ["pointerdown", "wheel"]) renderer.domElement.addEventListener(ev, () => { lastInteraction = performance.now(); }, { passive: true });

  // the camera glides between a few views while nobody is holding it
  const VIEWS = [
    { pos: [1.45, 2.95, 3.4], target: [0, 1.8, -0.3] },       // over the fly's shoulder, at the screen
    { pos: [3.1, 2.15, 1.0], target: [0, 1.75, -0.05] },       // from the side: fly, gaze, screen
    { pos: [-1.75, 1.95, -0.15], target: [0.1, 1.6, 0.45] },   // from beside the monitor, at the fly's face
    { pos: [0.35, 2.15, 1.9], target: [0, 1.98, -0.6] },       // close on the screen
  ].map(v => ({ pos: new THREE.Vector3(...v.pos), target: new THREE.Vector3(...v.target) }));
  let viewIndex = 0, viewSince = performance.now();
  camera.position.copy(VIEWS[0].pos); controls.target.copy(VIEWS[0].target);

  const composer = new THREE.EffectComposer(renderer);
  composer.addPass(new THREE.RenderPass(scene, camera));
  const bloom = new THREE.UnrealBloomPass(new THREE.Vector2(innerWidth, innerHeight), 0.32, 0.5, 0.93);
  composer.addPass(bloom);
  composer.addPass(new THREE.ShaderPass(THREE.GammaCorrectionShader));

  // ---------- lights ----------------------------------------------------
  scene.add(new THREE.HemisphereLight(0x8090c0, 0x20160c, 0.28));
  const lamp = new THREE.SpotLight(0xffe0b0, 1.6, 12, Math.PI / 5, 0.6, 1.3);
  lamp.position.set(1.7, 3.6, 1.2); lamp.target.position.set(0, 1.0, 0); lamp.castShadow = true;
  lamp.shadow.mapSize.set(2048, 2048); lamp.shadow.bias = -0.0004; lamp.shadow.radius = 4;
  scene.add(lamp, lamp.target);
  const screenGlow = new THREE.PointLight(0x9fd0ff, 0.9, 4.5, 2);        // takes the colour of what is on screen
  screenGlow.position.set(0, 1.9, -0.1); scene.add(screenGlow);
  const rim = new THREE.PointLight(0x4c6cff, 0.45, 10, 2); rim.position.set(-3, 2.6, 2); scene.add(rim);

  // ---------- room ------------------------------------------------------
  const woodTex = srgb(canvasTex(512, 512, (x, w, h) => {
    x.fillStyle = "#3a2414"; x.fillRect(0, 0, w, h);
    for (let i = 0; i < 90; i++) {
      x.strokeStyle = `rgba(${20 + Math.random() * 30},${10 + Math.random() * 15},5,${0.15 + Math.random() * 0.2})`;
      x.lineWidth = 1 + Math.random() * 3; x.beginPath();
      const y0 = Math.random() * h; x.moveTo(0, y0);
      for (let s = 0; s <= w; s += 32) x.lineTo(s, y0 + Math.sin(s / 60 + i) * 6);
      x.stroke();
    }
  }));
  woodTex.wrapS = woodTex.wrapT = THREE.RepeatWrapping;
  const floorTex = woodTex.clone(); floorTex.needsUpdate = true; floorTex.repeat.set(6, 6);
  const floor = new THREE.Mesh(new THREE.PlaneGeometry(24, 24), new THREE.MeshStandardMaterial({ map: floorTex, color: 0x6a5a50, roughness: 0.9 }));
  floor.rotation.x = -Math.PI / 2; floor.receiveShadow = true; scene.add(floor);
  const wall = new THREE.Mesh(new THREE.PlaneGeometry(24, 8), new THREE.MeshStandardMaterial({ color: 0x14161f, roughness: 1 }));
  wall.position.set(0, 4, -1.6); wall.receiveShadow = true; scene.add(wall);

  // the desk
  const deskMat = new THREE.MeshStandardMaterial({ map: woodTex, roughness: 0.55, metalness: 0.05 });
  const top = new THREE.Mesh(new THREE.BoxGeometry(3.4, 0.08, 1.6), deskMat);
  top.position.set(0, 1.0, -0.2); top.castShadow = top.receiveShadow = true; scene.add(top);
  for (const [lx, lz] of [[-1.6, -0.9], [1.6, -0.9], [-1.6, 0.5], [1.6, 0.5]]) {
    const leg = new THREE.Mesh(new THREE.BoxGeometry(0.08, 1.0, 0.08), deskMat);
    leg.position.set(lx, 0.5, lz); leg.castShadow = true; scene.add(leg);
  }
  const DESK_Y = 1.04;

  // the monitor: bezel, stand, and the screen itself, which shows the game
  const SCREEN_W = 1.76, SCREEN_H = 1.32, SCREEN_C = new THREE.Vector3(0, 1.98, -0.62);
  const plastic = new THREE.MeshPhysicalMaterial({ color: 0x15161a, roughness: 0.35, metalness: 0.2, clearcoat: 0.6 });
  const bezel = new THREE.Mesh(new THREE.BoxGeometry(SCREEN_W + 0.12, SCREEN_H + 0.12, 0.07), plastic);
  bezel.position.copy(SCREEN_C).add(new THREE.Vector3(0, 0, -0.04)); bezel.castShadow = true; scene.add(bezel);
  const neck = new THREE.Mesh(new THREE.BoxGeometry(0.12, 0.34, 0.06), plastic); neck.position.set(0, DESK_Y + 0.18, -0.72); scene.add(neck);
  const foot = new THREE.Mesh(new THREE.BoxGeometry(0.62, 0.03, 0.34), plastic); foot.position.set(0, DESK_Y + 0.015, -0.66); foot.castShadow = true; scene.add(foot);
  const screenCanvas = document.createElement("canvas"); screenCanvas.width = 640; screenCanvas.height = 480;
  const screenCtx = screenCanvas.getContext("2d");
  screenCtx.fillStyle = "#101418"; screenCtx.fillRect(0, 0, 640, 480);
  const screenTex = srgb(new THREE.CanvasTexture(screenCanvas));
  const screen = new THREE.Mesh(new THREE.PlaneGeometry(SCREEN_W, SCREEN_H), new THREE.MeshBasicMaterial({ map: screenTex, color: 0xd8d8d8, toneMapped: false }));
  screen.position.copy(SCREEN_C); scene.add(screen);
  const screenPoint = (x, y) => new THREE.Vector3(SCREEN_C.x + (x / 640 - 0.5) * SCREEN_W, SCREEN_C.y + (0.5 - y / 480) * SCREEN_H, SCREEN_C.z + 0.005);

  // a mouse on a pad, under the fly's right front foot
  const pad = new THREE.Mesh(new THREE.BoxGeometry(0.5, 0.008, 0.4), new THREE.MeshStandardMaterial({ color: 0x1d2433, roughness: 0.95 }));
  pad.position.set(0.52, DESK_Y + 0.004, 0.42); pad.receiveShadow = true; scene.add(pad);
  const mouse3d = new THREE.Group();
  const mouseBody = new THREE.Mesh(new THREE.SphereGeometry(0.07, 20, 14), new THREE.MeshPhysicalMaterial({ color: 0xdfe2ea, roughness: 0.3, clearcoat: 0.8 }));
  mouseBody.scale.set(0.8, 0.45, 1.25); mouseBody.castShadow = true; mouse3d.add(mouseBody);
  const mouseLed = new THREE.Mesh(new THREE.BoxGeometry(0.004, 0.004, 0.05), new THREE.MeshBasicMaterial({ color: 0x5adcff }));
  mouseLed.position.set(0, 0.032, -0.02); mouse3d.add(mouseLed);
  mouse3d.position.set(0.52, DESK_Y + 0.03, 0.42); scene.add(mouse3d);
  // a keyboard, for scale
  const keyboard = new THREE.Mesh(new THREE.BoxGeometry(0.9, 0.035, 0.3), plastic);
  keyboard.position.set(-0.6, DESK_Y + 0.018, 0.3); keyboard.castShadow = true; scene.add(keyboard);

  // ---------- the fly (the model from the card table) -------------------
  const wingTex = canvasTex(256, 128, (x, w, h) => {
    x.clearRect(0, 0, w, h); x.fillStyle = "rgba(220,232,255,0.55)";
    x.beginPath(); x.moveTo(0, h * 0.5); x.quadraticCurveTo(w * 0.35, 0, w * 0.98, h * 0.42);
    x.quadraticCurveTo(w * 0.9, h * 0.95, w * 0.3, h * 0.9); x.quadraticCurveTo(w * 0.05, h * 0.85, 0, h * 0.5); x.closePath(); x.fill();
    x.strokeStyle = "rgba(60,70,110,0.55)"; x.lineWidth = 1.6;
    for (const [c1, c2] of [[0.15, 0.25], [0.3, 0.42], [0.45, 0.58], [0.55, 0.74]]) {
      x.beginPath(); x.moveTo(6, h * 0.5); x.quadraticCurveTo(w * 0.5, h * c1, w * 0.95, h * c2); x.stroke();
    }
  });
  const wingShape = new THREE.Shape();
  wingShape.moveTo(0, 0); wingShape.quadraticCurveTo(0.24, 0.16, 0.66, 0.05); wingShape.quadraticCurveTo(0.6, -0.16, 0.2, -0.13); wingShape.quadraticCurveTo(0.03, -0.11, 0, 0);
  const wingGeo = new THREE.ShapeGeometry(wingShape, 12);
  { const pos = wingGeo.attributes.position, uv = wingGeo.attributes.uv;
    for (let i = 0; i < pos.count; i++) uv.setXY(i, pos.getX(i) / 0.66, (pos.getY(i) + 0.16) / 0.32); }

  function makeFly() {
    const g = new THREE.Group();
    const bodyMat = new THREE.MeshPhysicalMaterial({ color: 0xa86a36, roughness: 0.55, metalness: 0.05, clearcoat: 0.35, clearcoatRoughness: 0.5 });
    const abdMat = new THREE.MeshPhysicalMaterial({ color: 0x7a4a24, roughness: 0.6, metalness: 0.05, clearcoat: 0.3 });
    const darkMat = new THREE.MeshStandardMaterial({ color: 0x1a120c, roughness: 0.75 });
    const thorax = new THREE.Mesh(new THREE.SphereGeometry(0.22, 28, 20), bodyMat); thorax.scale.set(1, 0.92, 1.15); thorax.castShadow = true; g.add(thorax);
    for (let i = 0; i < 10; i++) {
      const b = new THREE.Mesh(new THREE.CylinderGeometry(0.003, 0.007, 0.09, 4), darkMat), a = (i / 10) * Math.PI * 2;
      b.position.set(Math.cos(a) * 0.14, 0.19, Math.sin(a) * 0.12); b.rotation.set(Math.sin(a) * 0.5, 0, -Math.cos(a) * 0.5); g.add(b);
    }
    const abdomen = new THREE.Mesh(new THREE.SphereGeometry(0.25, 28, 20), abdMat);
    abdomen.scale.set(0.92, 0.8, 1.6); abdomen.position.set(0, -0.03, -0.44); abdomen.castShadow = true; g.add(abdomen);
    for (let i = 0; i < 4; i++) {
      const stripe = new THREE.Mesh(new THREE.TorusGeometry(0.215 - i * 0.03, 0.02, 8, 36), darkMat);
      stripe.position.set(0, -0.03, -0.3 - i * 0.12); stripe.scale.set(0.95, 0.75, 1); g.add(stripe);
    }
    const head = new THREE.Group(); head.position.set(0, 0.05, 0.31); head.rotation.order = "YXZ"; g.add(head);
    const skull = new THREE.Mesh(new THREE.SphereGeometry(0.165, 28, 20), bodyMat); skull.scale.set(1.05, 1, 0.95); skull.castShadow = true; head.add(skull);
    const eyeMat = new THREE.MeshStandardMaterial({ color: 0xff2020, emissive: 0xff1010, emissiveIntensity: 0.85, roughness: 0.25 });
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
    // the proboscis: a pivot at the base, so it can swing forward and stretch when the fly presses
    const probPivot = new THREE.Group(); probPivot.position.set(0, -0.1, 0.06); head.add(probPivot);
    const proboscis = new THREE.Mesh(new THREE.CylinderGeometry(0.022, 0.04, 0.12, 10), darkMat); proboscis.position.y = -0.06; probPivot.add(proboscis);
    const labellum = new THREE.Mesh(new THREE.SphereGeometry(0.035, 12, 10), darkMat); labellum.position.y = -0.125; labellum.scale.set(1.3, 0.6, 1.1); probPivot.add(labellum);
    probPivot.rotation.x = 0.35;
    const legs = [];
    for (let i = 0; i < 3; i++) for (const s of [-1, 1]) {
      const hip = new THREE.Group(); hip.position.set(s * 0.15, -0.09, 0.13 - i * 0.15);
      const femur = new THREE.Mesh(new THREE.CylinderGeometry(0.014, 0.018, 0.28, 7), darkMat); femur.position.set(s * 0.12, -0.04, 0); femur.rotation.z = s * 1.15; hip.add(femur);
      const knee = new THREE.Group(); knee.position.set(s * 0.24, -0.09, 0); hip.add(knee);
      const tibia = new THREE.Mesh(new THREE.CylinderGeometry(0.008, 0.013, 0.3, 6), darkMat); tibia.position.set(s * 0.02, -0.15, 0); tibia.rotation.z = s * 0.15; knee.add(tibia);
      const tarsus = new THREE.Mesh(new THREE.CylinderGeometry(0.005, 0.008, 0.1, 5), darkMat); tarsus.position.set(s * 0.05, -0.32, 0.02); tarsus.rotation.z = s * 0.5; knee.add(tarsus);
      g.add(hip); legs.push({ hip, knee, side: s, pair: i });
    }
    const wingMat = new THREE.MeshPhysicalMaterial({ map: wingTex, transparent: true, opacity: 0.85, roughness: 0.1, metalness: 0.15,
      side: THREE.DoubleSide, clearcoat: 1, clearcoatRoughness: 0.05, depthWrite: false, color: 0xe8f0ff });
    const wings = [];
    for (const s of [-1, 1]) {
      const pivot = new THREE.Group(); pivot.position.set(s * 0.09, 0.2, -0.05);
      const w = new THREE.Mesh(wingGeo, wingMat); w.rotation.x = -Math.PI / 2; w.scale.set(s, 1, 1); w.rotation.z = s * 0.12; pivot.add(w);
      pivot.rotation.z = s * 0.2; g.add(pivot); wings.push({ pivot, side: s });
    }
    return { group: g, head, eyes, legs, wings, probPivot, phase: Math.random() * 10 };
  }

  const fly = makeFly();
  const FLY_SCALE = 1.0;
  fly.group.scale.setScalar(FLY_SCALE);
  fly.group.position.set(0, DESK_Y + 0.5, 0.62);
  fly.group.rotation.y = Math.PI;                        // the model faces +z; the monitor is at -z
  scene.add(fly.group);

  // the gaze: a faint beam from between the eyes to the spot on the screen
  const beamMat = new THREE.MeshBasicMaterial({ color: 0xff6a5a, transparent: true, opacity: 0.22, depthWrite: false, blending: THREE.AdditiveBlending });
  const beam = new THREE.Mesh(new THREE.CylinderGeometry(0.004, 0.012, 1, 8, 1, true), beamMat); scene.add(beam);
  const spotMat = new THREE.MeshBasicMaterial({ color: 0xff6a5a, transparent: true, opacity: 0.8, depthWrite: false, blending: THREE.AdditiveBlending });
  const gazeSpot = new THREE.Mesh(new THREE.RingGeometry(0.018, 0.028, 24), spotMat); scene.add(gazeSpot);

  function ripple(at, color) {
    const m = new THREE.Mesh(new THREE.RingGeometry(0.02, 0.032, 32),
      new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 1, depthWrite: false, blending: THREE.AdditiveBlending }));
    m.position.copy(at); m.position.z += 0.004; scene.add(m);
    tween(420, k => { m.scale.setScalar(1 + k * 3.5); m.material.opacity = 1 - k; }, { done: () => scene.remove(m) });
  }

  // ---------- the stream -----------------------------------------------
  let meta = null, owner = null, ow = 320, oh = 240;
  const gaze = { x: 320, y: 240, target: new THREE.Vector3().copy(SCREEN_C) };
  const state = { pressed: 0, lastPress: -1e9, wingsUntil: 0, lastEventT: -1, holding: null };
  const $ = id => document.getElementById(id);
  const panels = { eye: $("eye").getContext("2d"), sal: $("sal").getContext("2d"),
    layers: [0, 1, 2, 3].map(i => $("l" + i).getContext("2d")) };

  function paint(ctx, w, h, colourOf) {
    // each pixel takes its column's colour, through the eye's own patch map
    const img = ctx.createImageData(w, h), d = img.data, sx = ow / w, sy = oh / h;
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const k = owner[Math.floor(y * sy) * ow + Math.floor(x * sx)], o = (y * w + x) * 4;
      colourOf(k, d, o); d[o + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
  }
  function cross(ctx, x, y, s, colour) {
    ctx.strokeStyle = colour; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(x - s, y); ctx.lineTo(x + s, y); ctx.moveTo(x, y - s); ctx.lineTo(x, y + s); ctx.stroke();
    ctx.beginPath(); ctx.arc(x, y, s * 0.6, 0, Math.PI * 2); ctx.stroke();
  }
  const heat = v => [Math.min(255, v * 1.6), Math.max(0, v * 1.8 - 200), 40 + v * 0.3];           // dark -> red -> yellow
  const diverge = v => v < 128 ? [60 + v * 0.8, 110 + v * 0.9, 255] : [255, 255 - (v - 128) * 1.3, 240 - (v - 128) * 1.8];

  function onFrame(p) {
    $("connecting").style.display = "none";
    // the monitor
    const bytes = fromB64(p.screen);
    createImageBitmap(new Blob([bytes], { type: "image/jpeg" })).then(bmp => {
      screenCtx.drawImage(bmp, 0, 0); screenTex.needsUpdate = true; bmp.close && bmp.close();
    });
    // the panels
    const eye = fromB64(p.eye), sal = fromB64(p.salience);
    paint(panels.eye, 320, 240, (k, d, o) => { d[o] = eye[k * 3]; d[o + 1] = eye[k * 3 + 1]; d[o + 2] = eye[k * 3 + 2]; });
    cross(panels.eye, p.pointer[0] / 2, p.pointer[1] / 2, 9, "rgba(255,255,255,.9)");
    paint(panels.sal, 320, 240, (k, d, o) => { const c = heat(sal[k]); d[o] = c[0]; d[o + 1] = c[1]; d[o + 2] = c[2]; });
    cross(panels.sal, p.pointer[0] / 2, p.pointer[1] / 2, 9, "rgba(120,220,255,.95)");
    p.layers.forEach((s, i) => {
      const v = fromB64(s);
      paint(panels.layers[i], 160, 120, (k, d, o) => { const c = diverge(v[k]); d[o] = c[0]; d[o + 1] = c[1]; d[o + 2] = c[2]; });
    });
    // the average colour on screen lights the fly
    let r = 0, gg = 0, b = 0;
    for (let k = 0; k < eye.length; k += 3) { r += eye[k]; gg += eye[k + 1]; b += eye[k + 2]; }
    const n = eye.length / 3; screenGlow.color.setRGB(r / n / 255, gg / n / 255, b / n / 255);
    // gaze, presses, wings
    gaze.x = p.pointer[0]; gaze.y = p.pointer[1]; gaze.target.copy(screenPoint(gaze.x, gaze.y));
    if (p.presses > state.pressed) {
      const t = performance.now();
      if (p.last_event && p.last_event.kind === "wings" && p.last_event.t !== state.lastEventT) state.wingsUntil = t + 900;
      else { state.lastPress = t; ripple(gaze.target, p.last_event && p.last_event.kind === "place" ? 0x6ff0a8 : 0xffe084); }
      state.lastEventT = p.last_event ? p.last_event.t : -1;
    }
    state.pressed = p.presses; state.holding = p.holding;
    // numbers and the feed
    $("s-round").textContent = p.round; $("s-money").textContent = p.money; $("s-lives").textContent = Math.max(0, p.lives);
    $("s-towers").textContent = p.towers; $("lives-box").classList.toggle("low", p.lives <= 10); $("gen-n").textContent = p.generation;
    for (const bt of document.querySelectorAll("[data-speed]")) bt.classList.toggle("on", Number(bt.dataset.speed) === p.speed);
    // the feed, newest first, with runs of the same event folded into one line
    const runs = [];
    for (const e of p.events.slice().reverse()) {
      const last = runs[runs.length - 1];
      if (last && last.text === e.text) last.n++; else runs.push({ ...e, n: 1 });
    }
    const ul = $("events"); ul.innerHTML = "";
    for (const e of runs.slice(0, 6)) {
      const li = document.createElement("li"); li.className = e.kind;
      const m = Math.floor(e.t / 60), s = (e.t % 60).toFixed(1).padStart(4, "0");
      li.innerHTML = `<span class="t">${m}:${s}</span><span class="x">${e.text}${e.n > 1 ? ` <small>×${e.n}</small>` : ""}</span>`;
      ul.appendChild(li);
    }
    const toast = $("toast");
    if (p.ended) {
      toast.textContent = p.won ? `ניצח! עם ${p.lives} חיים` : `הפסיד בסיבוב ${p.round}`;
      toast.className = "show " + (p.won ? "win" : "lose");
    } else toast.className = "";
  }

  fetch("/meta").then(r => r.json()).then(m => {
    meta = m; ow = m.owner_size[0]; oh = m.owner_size[1];
    owner = new Uint16Array(fromB64(m.owner).buffer);
    m.layers.forEach((label, i) => { $("c" + i).textContent = label; });
    $("subtitle").textContent = `FlyWire · אונה אופטית ימנית · ${m.neurons.toLocaleString()} נוירונים · ${m.columns} עמודות`;
    const src = new EventSource("/events");
    src.onmessage = e => onFrame(JSON.parse(e.data));
  });
  for (const bt of document.querySelectorAll("[data-speed]"))
    bt.onclick = () => fetch("/speed", { method: "POST", body: JSON.stringify({ speed: Number(bt.dataset.speed) }) });
  $("new").onclick = () => fetch("/new", { method: "POST", body: "{}" });

  // ---------- animation -------------------------------------------------
  const clock = new THREE.Clock();
  const tmp = new THREE.Vector3(), eyeWorld = new THREE.Vector3(), up = new THREE.Vector3(0, 1, 0);
  let headYaw = 0, headPitch = 0;
  function animate() {
    requestAnimationFrame(animate);
    const now = performance.now(), dt = Math.min(clock.getDelta(), 0.05), t = clock.elapsedTime;
    runTweens(now);
    const g = fly.group;
    // the head turns to the gaze: a saccade is fast, so the head is quick to follow
    tmp.copy(gaze.target); g.worldToLocal(tmp); tmp.sub(fly.head.position);
    const yaw = Math.max(-0.75, Math.min(0.75, Math.atan2(tmp.x, tmp.z)));
    const pitch = Math.max(-0.7, Math.min(0.5, -Math.atan2(tmp.y, Math.hypot(tmp.x, tmp.z))));
    headYaw += (yaw - headYaw) * 0.45; headPitch += (pitch - headPitch) * 0.45;
    fly.head.rotation.y = headYaw; fly.head.rotation.x = headPitch;
    g.rotation.y = Math.PI + headYaw * 0.25;
    g.position.y = DESK_Y + 0.5 + 0.012 * Math.sin(t * 2.1 + fly.phase);
    // the proboscis reaches out on a press
    const since = (now - state.lastPress) / 1000, reach = since < 0.35 ? Math.sin(Math.min(1, since / 0.35) * Math.PI) : 0;
    fly.probPivot.rotation.x = 0.35 - reach * 1.1; fly.probPivot.scale.y = 1 + reach * 1.4;
    mouse3d.position.y = DESK_Y + 0.03 - reach * 0.012;
    // the mouse on its pad follows the pointer, a little
    mouse3d.position.x += ((0.52 + (gaze.x / 640 - 0.5) * 0.3) - mouse3d.position.x) * 0.3;
    mouse3d.position.z += ((0.42 + (gaze.y / 480 - 0.5) * 0.22) - mouse3d.position.z) * 0.3;
    mouseLed.material.color.setHex(reach > 0.05 ? 0xffe084 : 0x5adcff);
    // wings: a burst when the fly starts a round, a slow tremor otherwise
    const buzzing = now < state.wingsUntil;
    for (const w of fly.wings) {
      const beat = buzzing ? Math.sin(t * 60 + fly.phase) * 0.7 : 0.04 * Math.sin(t * 2.4 + fly.phase);
      w.pivot.rotation.z = w.side * (0.2 + beat); w.pivot.rotation.y = w.side * (buzzing ? -0.25 : 0);
    }
    fly.head.userData.antL.rotation.x = -0.5 + 0.08 * Math.sin(t * 9 + fly.phase);
    fly.head.userData.antR.rotation.x = -0.5 + 0.08 * Math.sin(t * 11);
    for (const L of fly.legs) {
      const base = L.pair === 0 ? 0.35 : L.pair === 1 ? 0 : -0.35;
      L.hip.rotation.x = base + 0.025 * Math.sin(t * 6 + L.pair * 1.3 + (L.side < 0 ? 0 : 1));
    }
    // the beam from between the eyes to the gaze spot
    eyeWorld.set(0, 0.04, 0.16); fly.head.localToWorld(eyeWorld);
    const len = eyeWorld.distanceTo(gaze.target);
    beam.position.copy(eyeWorld).lerp(gaze.target, 0.5); beam.scale.set(1, len, 1);
    beam.quaternion.setFromUnitVectors(up, tmp.copy(gaze.target).sub(eyeWorld).normalize());
    beamMat.opacity = 0.16 + reach * 0.5;
    gazeSpot.position.copy(gaze.target); gazeSpot.scale.setScalar(1 + reach * 0.8);
    // the camera follows unless someone is holding it
    if (now - lastInteraction > 6000) {
      if (now - viewSince > 20000) { viewIndex = (viewIndex + 1) % VIEWS.length; viewSince = now; }
      const v = VIEWS[viewIndex], k = 1 - Math.exp(-dt * 0.7);
      camera.position.lerp(v.pos, k); controls.target.lerp(v.target, k);
    } else viewSince = now;
    controls.update();
    composer.render();
  }
  animate();

  addEventListener("resize", () => {
    camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix();
    renderer.setSize(innerWidth, innerHeight); composer.setSize(innerWidth, innerHeight);
  });
})();
