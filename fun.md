---
layout: default
title: Fun
permalink: /fun/
---

<article class="post projects">
    <header>
        <h1>Fun</h1>
        <p class="tagline">Projects built purely for the joy of it.</p>
    </header>

    <div class="project">
        <h3><a href="{{ '/posts/14/the-list/' | relative_url }}">The List</a> <span class="cv-year">2026 – Present</span></h3>
        <p class="project-kicker">An essential reference for the modern professional</p>
        <img src="{{ '/assets/images/posts/the-list-header.jpg' | relative_url }}" alt="A woodcut engraving of a stern bearded scholar recording forbidden words with a quill">
        <p>A rare and historically significant archive of forbidden words and forbidden places, assembled at considerable personal risk and held under active curation. Includes an interactive atlas of fieldwork sites. Discretion is advised.</p>
    </div>

    <div class="project">
        <h3>Mesh.ByFace <span class="cv-year">2019</span></h3>
        <p class="project-kicker">Turning Dynamo into a selfie machine, with the Hackstreet Boys</p>
        <img src="{{ '/assets/images/projects/mesh-byface.jpg' | relative_url }}" alt="The Mesh.ByFace pipeline: four coloured 3D face meshes, the webcam node in Dynamo, a landmark-tracked selfie and the point-cloud graph">
        <p>At the inaugural Dynamo Hackathon, I organised a team from four different architectural practices — Ben Robinson (Hawkins\Brown), Mauro Sabiu (Zaha Hadid Architects), Mikael Santrolli (Foster + Partners) and me (AHMM). We called ourselves the Hackstreet Boys, and our goal was to build something technologically ambitious while having as much fun as legally possible.</p>
        <p>The pipeline, built in one caffeinated sprint with help from Luke and Keith of the Dynamo development team: a live webcam node written in C# with AForge.NET, PyTorch face-alignment extracting 68 facial landmarks from your selfie, a morphable face model reshaping a neutral 3D mesh to match, pixel colours mapped onto the mesh triangles — and a Tweepy node live-tweeting the result, because obviously.</p>
        <p>We came third — and won a special prize for being the most fun team at the event. The Dynamo team published <a href="https://dynamobim.org/london-hackathon-hackstreet-boys/">a write-up of the whole glorious mess</a>. #BIMSelfie</p>
    </div>

    <div class="project">
        <h3>Video Games Design <span class="cv-year">2005 – 2008</span></h3>
        <p class="project-kicker">Flash games sold to American &amp; Canadian gaming companies</p>
        <img src="{{ '/assets/images/projects/flash-games.jpg' | relative_url }}" alt="Menu screen of The Ultimate Gamer Challenge, one of around twenty Flash games">
        <p>As a teenager I built online Flash games with my brother — he wrote all the ActionScript, I handled creative vision, graphics and animation.</p>
        <p>We found an audience and ended up selling our games to several American and Canadian gaming companies — a hobby that quietly turned into a small business with full creative control.</p>
        <div class="project-pair">
            <img src="{{ '/assets/images/projects/flash-pirates.jpg' | relative_url }}" alt="Pirates: a gold-lettered menu screen with a galleon on the high seas">
            <img src="{{ '/assets/images/projects/flash-santa.jpg' | relative_url }}" alt="Santa In Hell: Father Christmas wreathed in flame on the title screen">
        </div>
        <div class="project-pair">
            <img src="{{ '/assets/images/projects/flash-rocketcar.jpg' | relative_url }}" alt="Rocketcar 2 mid-level: a pink car leaping factory chimneys against a blue sky">
            <img src="{{ '/assets/images/projects/flash-hansjurgen.jpg' | relative_url }}" alt="Hansjürgen, one of roughly twenty games from the family Flash workshop">
        </div>
        <p>Across three years and roughly twenty games — multiplayer leaderboards, antigravity simulations, robot opponents, upgrade systems — I picked up the rudiments of graphic design, animation and UI/UX. I still see gaming as the pinnacle of UX: it exists purely for its own joy.</p>
    </div>
</article>
