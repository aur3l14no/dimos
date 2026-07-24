{
  description = "Voxel ray tracing native module for DimOS";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    crane.url = "github:ipetkov/crane";
    dimos-rust = {
      url = "path:../../../../native/rust";
      flake = false;
    };
  };

  outputs =
    {
      crane,
      dimos-rust,
      flake-utils,
      nixpkgs,
      self,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs { inherit system; };
        craneLib = crane.mkLib pkgs;
        lib = pkgs.lib;
        python = pkgs.python312;

        dimosRustRoot = toString dimos-rust;
        dimosRustSrc = lib.cleanSourceWith {
          src = dimos-rust;
          name = "dimos-rust-source";
          filter =
            path: type:
            let
              rel = lib.removePrefix "${dimosRustRoot}/" (toString path);
              inCrate = crate: rel == crate || lib.hasPrefix "${crate}/" rel;
            in
            rel == "Cargo.toml"
            || (
              (inCrate "dimos-module" || inCrate "dimos-module-macros")
              && craneLib.filterCargoSources path type
            );
        };
        crateSrc = pkgs.lib.fileset.toSource {
          root = ./.;
          fileset = craneLib.fileset.commonCargoSources ./.;
        };
        wheelDistribution = "dimos_voxel_ray_tracing";
        wheelTag =
          "cp310-abi3-linux_${pkgs.stdenv.hostPlatform.uname.processor}";

        commonArgs = {
          pname = "voxel-ray-tracing";
          version = "0.1.0";
          src = crateSrc;

          cargoLock = ./Cargo.lock;
          cargoToml = ./Cargo.toml;
          env.PYO3_PYTHON = python.interpreter;
          nativeBuildInputs = [ python ];
          RUSTFLAGS = pkgs.lib.optionalString pkgs.stdenv.isDarwin (
            "-C link-arg=-undefined -C link-arg=dynamic_lookup"
          );
          postPatch = ''
            substituteInPlace Cargo.toml \
              --replace-fail '../../../../native/rust/dimos-module' '${dimosRustSrc}/dimos-module'
          '';
          strictDeps = true;
        };

        cargoArtifacts = craneLib.buildDepsOnly commonArgs;
        nativePackage = craneLib.buildPackage (
          commonArgs
          // {
            inherit cargoArtifacts;
            doCheck = false;
            postInstall = ''
              sitePackages=$out/${python.sitePackages}
              mkdir -p "$sitePackages"
              cp target/release/libdimos_voxel_ray_tracing${pkgs.stdenv.hostPlatform.extensions.sharedLibrary} \
                "$sitePackages/dimos_voxel_ray_tracing.abi3.so"
              PYTHONPATH="$sitePackages" ${python.interpreter} -c \
                "from dimos_voxel_ray_tracing import VoxelRayMapper"
            '';
            meta.mainProgram = "voxel_ray_tracing";
          }
        );
        wheelMetadata = pkgs.writeText "METADATA" ''
          Metadata-Version: 2.4
          Name: dimos-voxel-ray-tracing
          Version: ${commonArgs.version}
          Summary: Native Rust voxel-map module with raycast clearing for DimOS
          License-Expression: Apache-2.0
          Requires-Python: >=3.10
        '';
        wheelDescriptor = pkgs.writeText "WHEEL" ''
          Wheel-Version: 1.0
          Generator: Nix
          Root-Is-Purelib: false
          Tag: ${wheelTag}
        '';
        pythonWheel =
          assert pkgs.stdenv.hostPlatform.isLinux;
          pkgs.runCommand "dimos-voxel-ray-tracing-wheel-${commonArgs.version}"
            {
              nativeBuildInputs = [ python.pkgs.wheel ];
            }
            ''
              wheelRoot="$(mktemp -d)"
              distInfo="$wheelRoot/${wheelDistribution}-${commonArgs.version}.dist-info"
              mkdir -p "$distInfo" "$out/wheels" "$out/nix-support"
              cp \
                "${nativePackage}/${python.sitePackages}/dimos_voxel_ray_tracing.abi3.so" \
                "$wheelRoot/"
              cp "${wheelMetadata}" "$distInfo/METADATA"
              cp "${wheelDescriptor}" "$distInfo/WHEEL"
              wheel pack "$wheelRoot" --dest-dir "$out/wheels"
              ln -s "${nativePackage}" "$out/nix-support/native-package"
            '';
        package = pkgs.symlinkJoin {
          name = "voxel-ray-tracing-${commonArgs.version}";
          paths = [ nativePackage ] ++ lib.optional pkgs.stdenv.hostPlatform.isLinux pythonWheel;
          meta.mainProgram = "voxel_ray_tracing";
        };
        tests = craneLib.cargoTest (
          commonArgs
          // {
            inherit cargoArtifacts;
          }
        );
      in
      {
        packages =
          {
            default = package;
            native = nativePackage;
          }
          // lib.optionalAttrs pkgs.stdenv.hostPlatform.isLinux {
            python-wheel = pythonWheel;
          };
        checks = {
          build = package;
          inherit tests;
        };
      }
    );
}
