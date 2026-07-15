{
  description = "Voxel ray tracing native module for DimOS";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    crane.url = "github:ipetkov/crane";
    # Pure fallback for standalone `nix build path:.`; the runtime overrides
    # this input with native/rust from the current checkout.
    dimos-rust = {
      url = "github:dimensionalOS/dimos?dir=native/rust";
      flake = false;
    };
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
      crane,
      dimos-rust,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs { inherit system; };
        craneLib = crane.mkLib pkgs;
        # File sets require a path, while non-flake inputs expose a string-like outPath.
        dimosRustRoot = /. + builtins.unsafeDiscardStringContext dimos-rust.outPath;

        crateSrc = pkgs.lib.fileset.toSource {
          root = ./.;
          fileset = craneLib.fileset.commonCargoSources ./.;
        };

        dimosRustSrc = pkgs.lib.fileset.toSource {
          root = dimosRustRoot;
          fileset = pkgs.lib.fileset.unions [
            (craneLib.fileset.commonCargoSources (dimosRustRoot + "/dimos-module"))
            (craneLib.fileset.commonCargoSources (dimosRustRoot + "/dimos-module-macros"))
          ];
        };

        src = pkgs.runCommand "voxel-ray-tracing-src" { } ''
          mkdir -p $out/dimos/mapping/ray_tracing/rust
          cp -r ${crateSrc}/src $out/dimos/mapping/ray_tracing/rust/src
          cp ${crateSrc}/Cargo.toml $out/dimos/mapping/ray_tracing/rust/Cargo.toml
          cp ${crateSrc}/Cargo.lock $out/dimos/mapping/ray_tracing/rust/Cargo.lock

          mkdir -p $out/native/rust
          cp -r ${dimosRustSrc}/dimos-module $out/native/rust/dimos-module
          cp -r ${dimosRustSrc}/dimos-module-macros $out/native/rust/dimos-module-macros
        '';

        commonArgs = {
          pname = "voxel-ray-tracing";
          version = "0.1.0";

          inherit src;
          # Python extension modules resolve CPython symbols from the loading
          # interpreter on Darwin instead of linking libpython directly.
          RUSTFLAGS = pkgs.lib.optionalString pkgs.stdenv.hostPlatform.isDarwin
            "-C link-arg=-undefined -C link-arg=dynamic_lookup";
          cargoLock = ./Cargo.lock;
          cargoToml = ./Cargo.toml;
          postUnpack = ''
            cd $sourceRoot/dimos/mapping/ray_tracing/rust
            sourceRoot="."
          '';
        };

        cargoArtifacts = craneLib.buildDepsOnly commonArgs;
      in
      {
        packages.cargoArtifacts = cargoArtifacts;
        packages.default = craneLib.buildPackage (
          commonArgs
          // {
            inherit cargoArtifacts;
            cargoExtraArgs = "--locked --lib --bin voxel_ray_tracing";

            postInstall = ''
              extension="$out/lib/libdimos_voxel_ray_tracing${pkgs.stdenv.hostPlatform.extensions.sharedLibrary}"
              if [ ! -f "$extension" ]; then
                echo "missing PyO3 cdylib: $extension" >&2
                exit 1
              fi

              mkdir -p "$out/${pkgs.python312.sitePackages}"
              ln -s "../../$(basename "$extension")" \
                "$out/${pkgs.python312.sitePackages}/dimos_voxel_ray_tracing.abi3.so"
            '';

            meta.mainProgram = "voxel_ray_tracing";
          }
        );
      }
    );
}
