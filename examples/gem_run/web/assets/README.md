# Gem Run artwork

`world.png` is the decorative floating-island environment generated for this demo
with the built-in image-generation tool, using the original Gem Run visual
concept as a style reference. The bridge, robot, gems, rocks, and scoring remain
code-driven game objects. The background supplies no game state or decisions.

Generation prompt:

> Use case: stylized-concept. Asset type: background environment for a playable browser runner game. Image 1 is a STYLE AND ENVIRONMENT REFERENCE, not an image to copy as the whole game. Create a polished 3D game-art environment matching its beautiful floating islands, ancient ruins, lush foliage, waterfalls, warm lantern light, indigo night sky and cyan mist. Wide landscape 16:9 composition, camera looking forward across a deep floating-island valley, horizon near the upper quarter. Put detailed floating rock islands and ancient arches along the LEFT AND RIGHT EDGES, waterfalls descending into mist; leave the entire CENTRAL 65 percent and lower central foreground open blue mist/sky as negative space for a separately rendered three-lane stone bridge. Distant smaller islands may sit near the top horizon. Warm gold lanterns on edge islands, softly luminous blue atmosphere, rounded stylized stone and lush small plants, depth of field, high-end family adventure game look, visually consistent with the reference. ABSOLUTELY NO bridge, path, road, track, character, robot, gems, obstacles, interface, panels, signs, text, letters, words, logo, watermark or arrows anywhere. Background only, edge scenery must not intrude over the central playable area. It should feel like the same world as the reference, with detailed tactile surfaces and cinematic lighting.

The renderer embeds this asset in the generated HTML for offline playback.
