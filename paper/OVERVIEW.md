# PhishNet — project overview

## What it is

PhishNet is a phishing detector. You give it a web page, and it tells you whether
that page is trying to steal your login details.

The interesting part is not the detecting. Plenty of tools do that. The
interesting part is **how it decides where to concentrate its defences**, and the
answer is: by working out what each attack costs the attacker.

## The problem I started from

Take a phishing page that copies a bank's login screen. A good one is pixel for
pixel identical to the real thing, because the attacker literally copied it. So
asking "does this look like a bank?" gets you nowhere. A perfect copy passes that
test easily.

The question that actually works is different. Not "does this look like a bank",
but "does this look *exactly* like PayPal while sitting on a web address that
isn't PayPal's?" You can only answer that by reading the picture and the address
together. Either one alone tells you nothing.

That is the detection problem. But there is a second problem underneath it, and
that one bothered me more.

## The thing nobody was asking

When researchers build a defence against image attacks, they test it like this:
pick a fixed list of tricks an attacker might use, run all of them, report one
score. Higher score, better defence.

But a real attacker is not working down that list. They pick whatever is cheapest
to pull off. Some tricks cost almost nothing. Others are expensive, not in money,
but because they wreck the scam.

Blurring a logo, for example, is free and takes seconds, and the victim still
recognises the brand and still types their password. Deleting the logo entirely is
a different story. Now the victim cannot tell whose website they are on, so far
fewer of them fall for it, and the attacker earns less.

Those two attacks are not equally likely. But a defence trained the normal way
treats them as if they were. It spends just as much effort guarding the door
nobody uses as the one everybody uses.

## What I did

I gave every attack a price, scored on three things: what it costs in money, what
it costs in effort, and how many victims it scares off.

Then I trained the model with a budget. At each step, the attacker is only allowed
to use combinations they could actually afford. Out of 4,525 possible
combinations, about 1,345 fit inside the budget. The model rehearses against
those, so its effort lands on the attacks a real attacker would genuinely pick.

The detector itself reads a page three ways at once: the web address, what the
page looks like, and what the page says. Those three get combined *before* any
decision is made, rather than scored separately and averaged afterwards, because
the useful question is about how they relate to each other.

## What I found

I ran the whole thing 13 times. Every run used the same starting point, the same
data order and the same test attacks, so the only thing that differed between them
was the weighting.

Under attack, the cost-weighted model gets the right brand 78.2% of the time. The
model that defends every attack equally manages 76.8%. An undefended model gets
69.9%.

So the gain over defending everything equally is 1.4 points. Small, but it held up
in 11 of the 13 runs, and the odds of that being luck are about 6 in 1,000. Worth
saying plainly: at 5 runs this gap looked like nothing at all. It took 13 before I
could tell it was real.

The number I actually care about is a different one. Against logo deletion, my
model is **17 points worse** than the one that defends everything equally.

That is not a bug. That is the entire idea working. Logo deletion is expensive for
the attacker, so the model spent almost nothing defending it and moved that effort
onto the cheap attacks instead. The model that treats all attacks alike cannot
make that trade, because it has no notion that some doors matter more than others.

## What actually runs

It is a working system, not only a paper.

A Chrome extension scores the page you are actually looking at, which matters more
than it sounds: modern phishing kits detect scanners and serve them a clean,
harmless decoy, so checking the page from a server elsewhere can be fooled.
Checking it on your own screen cannot.

A warm verdict comes back in about 24 milliseconds. Sites already known to be safe
short-circuit in a fraction of that. Every verdict comes with a reason: which parts
of the web address looked wrong, and which part of the image matched a known brand.

## What it is built on

8,504 real phishing pages covering 103 brands, drawn from the Phishpedia dataset.
Pages were split so that no phishing kit appears on both sides of the split,
otherwise the model learns to recognise the kit instead of the brand.

## Where it falls short

Three things I would want a reader to know.

The prices I assigned to attacks are my own reasoned estimates. They are not
measured from real attacker behaviour, and nobody has that data publicly. If the
prices are wrong, the allocation is wrong with them.

The gain over defending everything equally is genuinely small. 1.4 points is real
and it is repeatable, but it is not dramatic, and I would rather say that than
dress it up.

And the results cover logo recognition on one dataset. Whether the same idea helps
on other kinds of attack, or other kinds of detector, is untested.

## The one-line version

Security is not about how much defence you have. It is about where you put it. You
never have the budget to guard every door, so the real question is which doors an
attacker would actually try, and whether that is where your effort went.
