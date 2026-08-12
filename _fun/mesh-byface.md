---
title: Mesh.ByFace
year: 2019
kicker: Turning Dynamo into a selfie machine, with the Hackstreet Boys
order: 2
image: /assets/images/projects/mesh-byface.jpg
---

<figure>

    <img src="{{ '/assets/images/projects/mesh-byface.jpg' | relative_url }}" alt="The Mesh.ByFace pipeline: four coloured 3D face meshes, the webcam node in Dynamo, a landmark-tracked selfie and the point-cloud graph">

    <figcaption>The full pipeline</figcaption>

</figure>
<p>At the inaugural Dynamo Hackathon, I organised a team from four different architectural practices—Ben Robinson (Hawkins\Brown), Mauro Sabiu (Zaha Hadid Architects), Mikael Santrolli (Foster + Partners) and me (AHMM). We called ourselves the Hackstreet Boys, and our goal was to build something technologically ambitious while having as much fun as legally possible.</p>
<p>The pipeline, built in one caffeinated sprint with help from Luke and Keith of the Dynamo development team: a live webcam node written in C# with AForge.NET, PyTorch face-alignment extracting 68 facial landmarks from your selfie, a morphable face model reshaping a neutral 3D mesh to match, pixel colours mapped onto the mesh triangles—and a Tweepy node live-tweeting the result, because obviously.</p>
<p>We came third—and won a special prize for being the most fun team at the event. The Dynamo team published <a href="https://dynamobim.org/london-hackathon-hackstreet-boys/">a write-up of the whole glorious mess</a>. #BIMSelfie</p>
