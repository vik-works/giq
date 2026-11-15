// SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
//
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useRef } from "react";
import {
  ACESFilmicToneMapping,
  Box3,
  DirectionalLight,
  PerspectiveCamera,
  PMREMGenerator,
  Scene,
  Vector3,
  WebGLRenderer,
} from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { RoomEnvironment } from "three/examples/jsm/environments/RoomEnvironment.js";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import "./MeshViewer.css";

/* The generated asset, turnable. Lit by an environment as well as a key
   light: the materials are PBR, and a metallic surface with nothing to
   reflect renders black. Drawn on demand, when the view or the size
   changes, not in a loop: a sandbox tab left open should cost nothing. */
export function MeshViewer({ glb, label }: { glb: Blob; label: string }) {
  const host = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = host.current;
    if (!el) return;
    const renderer = new WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.toneMapping = ACESFilmicToneMapping;
    el.appendChild(renderer.domElement);

    const scene = new Scene();
    const pmrem = new PMREMGenerator(renderer);
    const environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
    scene.environment = environment;
    const key = new DirectionalLight(undefined, 1.6);
    key.position.set(2, 3, 4);
    scene.add(key);

    const camera = new PerspectiveCamera(35, 1, 0.01, 100);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = false;
    const draw = () => renderer.render(scene, camera);
    controls.addEventListener("change", draw);

    const resize = () => {
      const { width, height } = el.getBoundingClientRect();
      if (!width || !height) return;
      renderer.setSize(width, height, false);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      draw();
    };
    const observer = new ResizeObserver(resize);
    observer.observe(el);

    let gone = false;
    void glb.arrayBuffer().then((buffer) => {
      if (gone) return;
      new GLTFLoader().parse(buffer, "", (gltf) => {
        if (gone) return;
        const box = new Box3().setFromObject(gltf.scene);
        const centre = box.getCenter(new Vector3());
        const radius = box.getSize(new Vector3()).length() / 2 || 1;
        gltf.scene.position.sub(centre);
        scene.add(gltf.scene);
        // Back far enough for the asset's bounding sphere, less the slack a
        // sphere round a box always has: an elongated prop would sit small.
        const distance = (radius / Math.sin((camera.fov * Math.PI) / 360)) * 0.82;
        camera.position.set(distance * 0.55, distance * 0.3, distance * 0.78);
        camera.near = distance / 100;
        camera.far = distance * 10;
        camera.updateProjectionMatrix();
        controls.target.set(0, 0, 0);
        controls.update();
        resize();
      });
    });

    return () => {
      gone = true;
      observer.disconnect();
      controls.dispose();
      environment.dispose();
      pmrem.dispose();
      scene.traverse((node) => {
        const mesh = node as { geometry?: { dispose(): void }; material?: unknown };
        mesh.geometry?.dispose();
        for (const material of [mesh.material].flat()) {
          const m = material as Record<string, { dispose?: () => void } | undefined> | undefined;
          if (!m) continue;
          for (const value of Object.values(m)) if (value?.dispose && value !== m) value.dispose();
        }
      });
      renderer.dispose();
      renderer.domElement.remove();
    };
  }, [glb]);

  return <div ref={host} className="pl-giq-3d-viewer" role="img" aria-label={label} />;
}
