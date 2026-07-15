{
  description = "Voxel ray tracing native module for DimOS";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    crane.url = "github:ipetkov/crane";
    dimos-rust = {
      # Pure fallback for standalone builds. The Python wrapper overrides this
      # input with native/rust from the current checkout.
      url = "github:dimensionalOS/dimos?dir=native/rust";
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
        python = pkgs.python312;

        dimosRustRoot = /. + builtins.unsafeDiscardStringContext dimos-rust.outPath;
        crateSrc = pkgs.lib.fileset.toSource {
          root = ./.;
          fileset = craneLib.fileset.commonCargoSources ./.;
        };
        dimosRustSrc = pkgs.lib.fileset.toSource {
          root = dimosRustRoot;
          fileset = craneLib.fileset.commonCargoSources dimosRustRoot;
        };
        src = pkgs.runCommand "voxel-ray-tracing-src" { } ''
          mkdir -p $out/dimos/mapping/ray_tracing/rust $out/native/rust
          cp -r ${crateSrc}/. $out/dimos/mapping/ray_tracing/rust/
          cp -r ${dimosRustSrc}/. $out/native/rust/
        '';

        commonArgs = {
          pname = "voxel-ray-tracing";
          version = "0.1.0";
          inherit src;

          cargoLock = ./Cargo.lock;
          cargoToml = ./Cargo.toml;
          RUSTFLAGS = pkgs.lib.optionalString pkgs.stdenv.isDarwin (
            "-C link-arg=-undefined -C link-arg=dynamic_lookup"
          );
          postUnpack = ''
            cd $sourceRoot/dimos/mapping/ray_tracing/rust
            sourceRoot=.
          '';
          strictDeps = true;
        };

        cargoArtifacts = craneLib.buildDepsOnly commonArgs;
        package = craneLib.buildPackage (
          commonArgs
          // {
            inherit cargoArtifacts;
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
      in
      {
        packages = {
          inherit cargoArtifacts;
          default = package;
        };
        checks.default = package;
      }
    );
}
