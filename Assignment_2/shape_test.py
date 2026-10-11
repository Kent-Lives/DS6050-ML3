"""
shape_test.py - quick shape checks for the CNN assignment.

Put this file in the same folder as VGGNet.py, NiN.py, GoogLeNet.py,
ResNet.py, MobileNet.py and transfer_learning.py, then run:

    python shape_test.py              # test everything that's implemented
    python shape_test.py vgg          # test only VGGNet.py
    python shape_test.py vgg nin      # test only those two
    python shape_test.py --transfer   # also test transfer_learning.py
                                      # (downloads pretrained weights, slow)

Groups: vgg, nin, googlenet, resnet, mobilenet, transfer

Nothing here trains anything. It builds each module, pushes one small fake
batch of images through it, and compares the output shape to what the
assignment says it should be. A run takes a couple of seconds.
"""

import sys
import traceback
from importlib import import_module

import torch


# ----------------------------------------------------------------------
# test harness
# ----------------------------------------------------------------------

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
TODO = "\033[93mTODO\033[0m"
SKIP = "\033[90mSKIP\033[0m"

results = {"pass": 0, "fail": 0, "todo": 0}


def check(label, build, input_shape, expected_shape):
    """Build a module, run one fake batch through it, compare shapes.

    `build` is a zero-argument function so that an error while *constructing*
    the module is caught here rather than crashing the whole script.
    """
    try:
        module = build()
    except NotImplementedError:
        print(f"  {TODO}  {label}  (__init__ not implemented yet)")
        results["todo"] += 1
        return
    except Exception:
        print(f"  {FAIL}  {label}  - error while building:")
        _print_last_error()
        results["fail"] += 1
        return

    module.eval()  # batch norm behaves differently in train mode on tiny batches
    x = torch.randn(*input_shape)

    try:
        with torch.no_grad():
            out = module(x)
    except NotImplementedError:
        print(f"  {TODO}  {label}  (forward not implemented yet)")
        results["todo"] += 1
        return
    except Exception:
        print(f"  {FAIL}  {label}  - error while running forward:")
        _print_last_error()
        results["fail"] += 1
        return

    got = tuple(out.shape)
    want = tuple(expected_shape)
    if got == want:
        print(f"  {PASS}  {label}  {tuple(input_shape)} -> {got}")
        results["pass"] += 1
    else:
        print(f"  {FAIL}  {label}  {tuple(input_shape)} -> {got}, expected {want}")
        _explain_shape_mismatch(got, want)
        results["fail"] += 1


def _print_last_error():
    """Print the deepest line of the traceback that is in the user's own code."""
    tb = traceback.format_exc().strip().splitlines()
    print(f"         {tb[-1]}")
    for line in reversed(tb):
        if ".py" in line and "shape_test.py" not in line and "site-packages" not in line:
            print(f"         {line.strip()}")
            break


def _explain_shape_mismatch(got, want):
    """Point at the most likely cause, since shape bugs are the common ones."""
    if len(got) != len(want):
        print(f"         Wrong number of dimensions. Missing a flatten, or flattening too early?")
        return
    names = ["batch", "channels", "height", "width"][: len(got)]
    for name, g, w in zip(names, got, want):
        if g != w:
            if name == "channels":
                print(f"         Channel count is off ({g} vs {w}). Check the out_channels "
                      f"of your convolutions and that each BatchNorm2d matches the conv before it.")
            elif name in ("height", "width"):
                print(f"         Spatial size is off ({g} vs {w}). Check padding and stride - "
                      f"kernel 3 wants padding 1, kernel 5 wants padding 2 to keep the size.")
            else:
                print(f"         The {name} dimension changed, which it never should.")
            return


def section(title):
    print(f"\n{title}")
    print("-" * len(title))


def load(module_name):
    """Import one of the assignment files, or report why we can't."""
    try:
        return import_module(module_name)
    except Exception:
        print(f"  {SKIP}  could not import {module_name}.py")
        _print_last_error()
        return None


# ----------------------------------------------------------------------
# the actual checks, one group per assignment file
# ----------------------------------------------------------------------

def test_vgg():
    section("VGGNet.py")
    m = load("VGGNet")
    if m is None:
        return
    # A VGG block keeps H and W through its convolutions (padding 1),
    # then halves them with 2x2 max pooling.
    check("VGGBlock(64, 128)", lambda: m.VGGBlock(64, 128),
          (2, 64, 32, 32), (2, 128, 16, 16))
    check("VGGBlock(64, 128, num_convs=3)", lambda: m.VGGBlock(64, 128, num_convs=3),
          (2, 64, 32, 32), (2, 128, 16, 16))
    # Two blocks means the image is halved twice: 150 -> 75 -> 37.
    check("VGGNet(num_classes=18)", lambda: m.VGGNet(num_classes=18),
          (2, 3, 150, 150), (2, 18))
    check("VGGNet(num_classes=5)", lambda: m.VGGNet(num_classes=5),
          (2, 3, 150, 150), (2, 5))


def test_nin():
    section("NiN.py")
    m = load("NiN")
    if m is None:
        return
    # A NiN block is one real convolution followed by two 1x1 convolutions,
    # so only the first convolution changes the spatial size.
    # floor((150 - 11) / 4) + 1 = 35
    check("NiNBlock(3, 96, 11, stride=4)", lambda: m.NiNBlock(3, 96, 11, stride=4),
          (2, 3, 150, 150), (2, 96, 35, 35))
    check("NiNBlock(96, 256, 5, padding=2)", lambda: m.NiNBlock(96, 256, 5, padding=2),
          (2, 96, 17, 17), (2, 256, 17, 17))
    check("NiN(num_classes=18)", lambda: m.NiN(num_classes=18),
          (2, 3, 150, 150), (2, 18))


def test_googlenet():
    section("GoogLeNet.py")
    m = load("GoogLeNet")
    if m is None:
        return
    # All four branches keep H and W, so the block only changes the channel
    # count: 64 + 128 + 32 + 32 = 256 for inception3a's settings.
    check("InceptionBlock(64, 64, 96, 128, 16, 32, 32)",
          lambda: m.InceptionBlock(64, 64, 96, 128, 16, 32, 32),
          (2, 64, 28, 28), (2, 256, 28, 28))
    # inception3b: 128 + 192 + 96 + 64 = 480.
    check("InceptionBlock(256, 128, 128, 192, 32, 96, 64)",
          lambda: m.InceptionBlock(256, 128, 128, 192, 32, 96, 64),
          (2, 256, 28, 28), (2, 480, 28, 28))
    check("GoogLeNet(num_classes=18)", lambda: m.GoogLeNet(num_classes=18),
          (2, 3, 150, 150), (2, 18))


def test_resnet():
    section("ResNet.py")
    m = load("ResNet")
    if m is None:
        return
    # Same in and out channels with stride 1: the shortcut is the identity.
    check("BasicBlock(64, 64)", lambda: m.BasicBlock(64, 64),
          (2, 64, 32, 32), (2, 64, 32, 32))
    # Changing channels and stride: the shortcut needs its 1x1 convolution,
    # otherwise the addition in forward will fail.
    check("BasicBlock(64, 128, stride=2)", lambda: m.BasicBlock(64, 128, stride=2),
          (2, 64, 32, 32), (2, 128, 16, 16))
    check("ResNet(num_classes=18)", lambda: m.ResNet(num_classes=18),
          (2, 3, 150, 150), (2, 18))


def test_mobilenet():
    section("MobileNet.py")
    m = load("MobileNet")
    if m is None:
        return
    check("DepthwiseSeparableConv(32, 64)", lambda: m.DepthwiseSeparableConv(32, 64),
          (2, 32, 56, 56), (2, 64, 56, 56))
    check("DepthwiseSeparableConv(32, 64, stride=2)",
          lambda: m.DepthwiseSeparableConv(32, 64, stride=2),
          (2, 32, 56, 56), (2, 64, 28, 28))
    # Stride 1 and equal channel counts, so this one uses the residual.
    check("InvertedResidual(32, 32, stride=1, expand_ratio=6)",
          lambda: m.InvertedResidual(32, 32, stride=1, expand_ratio=6),
          (2, 32, 56, 56), (2, 32, 56, 56))
    # Different channel counts, so no residual; the add would fail if you
    # used it here.
    check("InvertedResidual(32, 64, stride=2, expand_ratio=6)",
          lambda: m.InvertedResidual(32, 64, stride=2, expand_ratio=6),
          (2, 32, 56, 56), (2, 64, 28, 28))
    # expand_ratio 1 skips the expansion convolution entirely.
    check("InvertedResidual(32, 16, stride=1, expand_ratio=1)",
          lambda: m.InvertedResidual(32, 16, stride=1, expand_ratio=1),
          (2, 32, 56, 56), (2, 16, 56, 56))
    check("MobileNet(num_classes=18)", lambda: m.MobileNet(num_classes=18),
          (2, 3, 224, 224), (2, 18))
    check("MobileNet(num_classes=18, width_mult=0.5)",
          lambda: m.MobileNet(num_classes=18, width_mult=0.5),
          (2, 3, 224, 224), (2, 18))


def test_transfer():
    section("transfer_learning.py  (downloads pretrained weights the first time)")
    m = load("transfer_learning")
    if m is None:
        return

    for name, size in [("resnet18", 224), ("vgg16", 224), ("mobilenet_v2", 224)]:
        check(f"get_pretrained_model('{name}', feature_extract=True)",
              lambda name=name: m.get_pretrained_model(name, num_classes=18, feature_extract=True),
              (2, 3, size, size), (2, 18))

    # Freezing check: with feature_extract=True only the new final layer
    # should still be trainable.
    try:
        model = m.get_pretrained_model("resnet18", num_classes=18, feature_extract=True)
        trainable = [n for n, p in model.named_parameters() if p.requires_grad]
        frozen = [n for n, p in model.named_parameters() if not p.requires_grad]
        if frozen and all(n.startswith("fc") for n in trainable):
            print(f"  {PASS}  resnet18 freezing  ({len(frozen)} frozen, "
                  f"{len(trainable)} trainable: the new final layer)")
            results["pass"] += 1
        else:
            print(f"  {FAIL}  resnet18 freezing  - trainable parameters: {trainable[:6]}")
            print("         Expected only the replaced final layer to be trainable. "
                  "Freeze the parameters BEFORE replacing the final layer, since the "
                  "replacement arrives unfrozen.")
            results["fail"] += 1
    except NotImplementedError:
        print(f"  {TODO}  resnet18 freezing")
        results["todo"] += 1
    except Exception:
        print(f"  {FAIL}  resnet18 freezing  - error:")
        _print_last_error()
        results["fail"] += 1

    # Unsupported names should raise a ValueError.
    try:
        m.get_pretrained_model("not_a_real_model", num_classes=18)
        print(f"  {FAIL}  unsupported model name  - returned instead of raising ValueError")
        results["fail"] += 1
    except ValueError:
        print(f"  {PASS}  unsupported model name raises ValueError")
        results["pass"] += 1
    except NotImplementedError:
        print(f"  {TODO}  unsupported model name")
        results["todo"] += 1
    except Exception:
        print(f"  {FAIL}  unsupported model name  - raised the wrong kind of error:")
        _print_last_error()
        results["fail"] += 1


# ----------------------------------------------------------------------

GROUPS = {
    "vgg": test_vgg,
    "nin": test_nin,
    "googlenet": test_googlenet,
    "resnet": test_resnet,
    "mobilenet": test_mobilenet,
    "transfer": test_transfer,
}


def main():
    args = [a.lower() for a in sys.argv[1:]]
    include_transfer = "--transfer" in args
    chosen = [a for a in args if not a.startswith("-")]

    if chosen:
        unknown = [c for c in chosen if c not in GROUPS]
        if unknown:
            print(f"Unknown group(s): {', '.join(unknown)}")
            print(f"Available: {', '.join(GROUPS)}")
            return
        to_run = chosen
    else:
        to_run = [g for g in GROUPS if g != "transfer"]
        if include_transfer:
            to_run.append("transfer")

    torch.manual_seed(0)
    print("=" * 62)
    print("Shape checks")
    print("=" * 62)

    for group in to_run:
        GROUPS[group]()

    print("\n" + "=" * 62)
    print(f"{results['pass']} passed, {results['fail']} failed, "
          f"{results['todo']} not implemented yet")
    if not chosen and not include_transfer:
        print("transfer_learning.py was skipped; add --transfer to include it.")
    print("=" * 62)


if __name__ == "__main__":
    main()
